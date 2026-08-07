-- KEYS[1] = redis key for the run
-- ARGV[1] = job name being claimed
-- ARGV[2] = current unix time (seconds)
-- ARGV[3] = stale_after (seconds)
-- ARGV[4] = record TTL (seconds)
-- returns: {claimed(0/1), status, job_name, attempts}

local existing = redis.call('HGETALL', KEYS[1])
if #existing == 0 then
    redis.call('HSET', KEYS[1], 'status', 'running', 'job_name', ARGV[1], 'attempts', 1, 'updated_at', ARGV[2])
    redis.call('EXPIRE', KEYS[1], ARGV[4])
    return {1, 'running', ARGV[1], 1}
end

local rec = {}
for i = 1, #existing, 2 do rec[existing[i]] = existing[i + 1] end

if rec.status == 'running' and (tonumber(ARGV[2]) - tonumber(rec.updated_at)) < tonumber(ARGV[3]) then
    return {0, rec.status, rec.job_name, tonumber(rec.attempts)}
end

local attempts = tonumber(rec.attempts) + 1
redis.call('HSET', KEYS[1], 'status', 'running', 'job_name', ARGV[1], 'attempts', attempts, 'updated_at', ARGV[2])
redis.call('EXPIRE', KEYS[1], ARGV[4])
return {1, 'running', ARGV[1], attempts}
