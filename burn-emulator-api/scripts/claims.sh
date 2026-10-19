# sourced after env.sh; claim helpers over gcloud storage
# the claim's generation is the API's (the runner's claim_generation); these update metadata, which moves the metageneration
# return codes: 0 ok, 1 precondition failed or object missing, 2 other error

# gcloud storage <args>, stdout kept; stderr mapped to the return codes above
gcs () {
    local err rc
    err=$(mktemp)
    gcloud storage "$@" 2>"$err"
    rc=$?
    if [ "$rc" -ne 0 ]; then
        if grep -qiE 'precondition|conditionNotMet|412|matched no objects|not found|404' "$err"; then
            rc=1
        else
            cat "$err" >&2
            rc=2
        fi
    fi
    rm -f "$err"
    return "$rc"
}

# one JSON line per claim: {key, generation, metageneration, metadata}
claim_list () {
    local out err rc
    err=$(mktemp)
    out=$(gcloud storage objects list "$CLAIMS_URI/**" --raw --format=json 2>"$err")
    rc=$?
    if [ "$rc" -ne 0 ]; then
        # an empty _claims/ is a normal pass; any other failure (e.g. a wrong or missing bucket) is an error
        if grep -qi 'matched no objects' "$err" && ! grep -qiE 'bucket|404|403|permission' "$err"; then
            rm -f "$err"
            return 0
        fi
        cat "$err" >&2
        rm -f "$err"
        return 2
    fi
    rm -f "$err"
    jq -c '.[] | {key: (.name | ltrimstr("_claims/")), generation, metageneration, metadata: (.metadata // {})}' <<<"$out"
}

# claim_get <key>: the same shape as claim_list
claim_get () {
    local out
    out=$(gcs objects describe "$CLAIMS_URI/$1" --raw --format=json) || return
    jq -c '{key: (.name | ltrimstr("_claims/")), generation, metageneration, metadata: (.metadata // {})}' <<<"$out"
}

# claim_body <key> <generation>: the runner env JSON the API stored with the claim
claim_body () {
    gcs cat "$CLAIMS_URI/$1#$2"
}

# claim_update <key> <generation> <metageneration> [k=v | -k]...: set k=v, remove -k
claim_update () {
    local key=$1 gen=$2 mgen=$3
    shift 3
    local sets=() removes=() arg
    for arg in "$@"; do
        case $arg in
            -*) removes+=("${arg#-}") ;;
            *) sets+=("${arg//|/ }") ;;
        esac
    done
    local args=(--if-generation-match="$gen" --if-metageneration-match="$mgen")
    if [ ${#sets[@]} -gt 0 ]; then
        args+=("--update-custom-metadata=^|^$(IFS='|'; echo "${sets[*]}")")
    fi
    if [ ${#removes[@]} -gt 0 ]; then
        args+=("--remove-custom-metadata=$(IFS=,; echo "${removes[*]}")")
    fi
    gcs objects update "$CLAIMS_URI/$key" "${args[@]}" >/dev/null
}

# DL or PT from the job id's hash suffix (dispatch.JobID.Backend)
claim_backend () {
    if [[ ${1##*/} == *1 ]]; then echo PT; else echo DL; fi
}
