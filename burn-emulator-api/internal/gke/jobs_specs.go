package k8s

import (
	"context"
	"fmt"
	"strings"

	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
)

type Config struct {
	Namespace      string
	ServiceAccount string
	ImageStore     string
	OutputBucket   string
}

type Client struct {
	clientset *kubernetes.Clientset
	cfg       Config
}

func NewClient(cfg Config) (*Client, error) {
	restCfg, err := rest.InClusterConfig()
	if err != nil {
		return nil, fmt.Errorf("loading in-cluster config: %w", err)
	}
	cs, err := kubernetes.NewForConfig(restCfg)
	if err != nil {
		return nil, fmt.Errorf("creating clientset: %w", err)
	}
	return &Client{clientset: cs, cfg: cfg}, nil
}

type JobRequest struct {
	Variation string
	Caching   string
	JobName   string
	FuelsPath string
}

func (c *Client) CreateJob(ctx context.Context, req JobRequest) (string, error) {
	name := fmt.Sprintf("%s-%s", req.JobName, rand.String(8))
	job := buildJobSpec(name, c.cfg, req)

	created, err := c.clientset.BatchV1().Jobs(c.cfg.Namespace).Create(ctx, job, metav1.CreateOptions{})
	if err != nil {
		return "", fmt.Errorf("creating job: %w", err)
	}
	return created.Name, nil
}

func buildJobSpec(jobName string, cfg Config, req JobRequest) *batchv1.Job {
	backoffLimit := int32(1)
	ttl := int32(3600)
	runAsNonRoot := true
	runAsUser := int64(1000)
	allowPrivEsc := false
	readOnlyRootFS := true
	outPath := fmt.Sprintf("%s/%s", strings.TrimSuffix(cfg.OutputBucket, "/"), req.Variation)

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
								{Name: "BURN_EMULATOR_CACHING", Value: req.Caching},
								{Name: "BURN_EMULATOR_OUT_PATH", Value: outPath},
								{Name: "BURN_EMULATOR_RUN_NAME", Value: jobName},
								{Name: "BURN_EMULATOR_FUELS_PATH", Value: req.FuelsPath},
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
		panic(err) // only ever called with hard-coded valid literals above
	}
	return q
}
