package k8s

import (
	"context"
	"fmt"
	"strings"

	"github.com/redis/go-redis/v9"

	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"

	storage "google.golang.org/api/storage/v1"
)

type Config struct {
	Namespace      string
	ServiceAccount string
	ImageStore     string
	OutputBucket   string
	RedisAddr      string
}

type Client struct {
	clientset *kubernetes.Clientset
	storage   *storage.Service
	redis     *redis.Client
	cfg       Config
}

func NewClient(ctx context.Context, cfg Config) (*Client, error) {
	restCfg, err := rest.InClusterConfig()
	if err != nil {
		return nil, fmt.Errorf("loading in-cluster config: %w", err)
	}
	cs, err := kubernetes.NewForConfig(restCfg)
	if err != nil {
		return nil, fmt.Errorf("creating clientset: %w", err)
	}
	storageSvc, err := storage.NewService(ctx)
	if err != nil {
		return nil, fmt.Errorf("creating storage client: %w", err)
	}
	redisClient := redis.NewClient(&redis.Options{Addr: cfg.RedisAddr})
	if err := redisClient.Ping(ctx).Err(); err != nil {
		return nil, fmt.Errorf("connecting to redis at %s: %w", cfg.RedisAddr, err)
	}
	return &Client{clientset: cs, storage: storageSvc, redis: redisClient, cfg: cfg}, nil
}

type JobRequest struct {
	TreatmentArea   string
	TreatmentBuff   float32
	TreatmentSeed   float32
	IgnitionDensity float32
	Variation       string
	JobName         string
}

type CreateJobResult struct {
	JobName    string // "cached" | "pending" | "scheduled"
	Status     string
	Attempts   int
	OutputPath string
}

func (c *Client) CreateJob(ctx context.Context, req JobRequest) (CreateJobResult, error) {
	key := CacheKey(req)
	outPath := fmt.Sprintf("%s/%s/%s", strings.TrimSuffix(c.cfg.OutputBucket, "/"), req.Variation, key)

	cached, err := c.outputExists(ctx, outPath)
	if err != nil {
		return CreateJobResult{}, fmt.Errorf("checking output cache: %w", err)
	}
	if cached {
		return CreateJobResult{Status: "cached", OutputPath: outPath}, nil
	}

	name := fmt.Sprintf("%s-%s", req.JobName, key[:8])

	claimed, existing, err := c.claimRun(ctx, key, name)
	if err != nil {
		return CreateJobResult{}, fmt.Errorf("claiming run: %w", err)
	}
	if !claimed {
		return CreateJobResult{JobName: existing.JobName, Status: "pending", Attempts: existing.Attempts, OutputPath: outPath}, nil
	}

	job := buildJobSpec(name, c.cfg, req, outPath)

	created, err := c.clientset.BatchV1().Jobs(c.cfg.Namespace).Create(ctx, job, metav1.CreateOptions{})
	if err != nil {
		c.releaseRun(ctx, key)
		return CreateJobResult{}, fmt.Errorf("creating job: %w", err)
	}
	return CreateJobResult{JobName: created.Name, Status: "scheduled", Attempts: existing.Attempts, OutputPath: outPath}, nil
}

func buildJobSpec(jobName string, cfg Config, req JobRequest, outPath string) *batchv1.Job {
	backoffLimit := int32(1)
	ttl := int32(3600)
	runAsNonRoot := true
	runAsUser := int64(1000)
	allowPrivEsc := false
	readOnlyRootFS := true

	return &batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{
			Name:      jobName,
			Namespace: cfg.Namespace,
			Labels:    map[string]string{"app": "burn-emulator"},
		},
		Spec: batchv1.JobSpec{
			BackoffLimit:            &backoffLimit,
			TTLSecondsAfterFinished: &ttl,
			Template: corev1.PodTemplateSpec{
				Spec: corev1.PodSpec{
					ServiceAccountName: cfg.ServiceAccount,
					RestartPolicy:      corev1.RestartPolicyNever,
					SecurityContext: &corev1.PodSecurityContext{
						RunAsNonRoot: &runAsNonRoot,
						RunAsUser:    &runAsUser,
					},
					Containers: []corev1.Container{
						{
							Name:  "runner",
							Image: fmt.Sprintf("%s:%s", cfg.ImageStore, req.Variation),
							Env: []corev1.EnvVar{
								{Name: "BURN_EMULATOR_VARIATION", Value: req.Variation},
								{Name: "BURN_EMULATOR_TREATMENT_AREA", Value: req.TreatmentArea},
								{Name: "BURN_EMULATOR_TREATMENT_BUFF", Value: fmt.Sprintf("%f", req.TreatmentBuff)},
								{Name: "BURN_EMULATOR_TREATMENT_SEED", Value: fmt.Sprintf("%f", req.TreatmentSeed)},
								{Name: "BURN_EMULATOR_IGNITION_DENSITY", Value: fmt.Sprintf("%f", req.IgnitionDensity)},
								{Name: "BURN_EMULATOR_OUT_PATH", Value: outPath},
								{Name: "BURN_EMULATOR_RUN_NAME", Value: jobName},
							},
							SecurityContext: &corev1.SecurityContext{
								AllowPrivilegeEscalation: &allowPrivEsc,
								ReadOnlyRootFilesystem:   &readOnlyRootFS,
							},
							Resources: corev1.ResourceRequirements{
								Limits: corev1.ResourceList{
									// TODO: spec for cheapest spot instance GPUs &/or TPUs
									corev1.ResourceCPU:    mustQty("500m"),
									corev1.ResourceMemory: mustQty("512Mi"),
								},
							},
						},
					},
				},
			},
		},
	}
}

func mustQty(s string) resource.Quantity {
	q, err := resource.ParseQuantity(s)
	if err != nil {
		panic(err)
	}
	return q
}
