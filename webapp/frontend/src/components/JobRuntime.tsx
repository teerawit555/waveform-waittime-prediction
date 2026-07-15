import { Clock3 } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import { JobResponse } from '../lib/api';

type JobRuntimeProps = {
  job?: Partial<JobResponse> | null;
  label: string;
};

function parseTimestamp(value?: string | null) {
  if (!value) return null;
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? parsed : null;
}

export function formatJobDuration(totalSeconds: number) {
  const seconds = Math.round(Math.max(0, totalSeconds) * 10) / 10;
  if (seconds < 60) return `${seconds.toFixed(1)} sec`;

  const wholeSeconds = Math.floor(seconds);
  const hours = Math.floor(wholeSeconds / 3600);
  const minutes = Math.floor((wholeSeconds % 3600) / 60);
  const remainder = (seconds % 60).toFixed(1).padStart(4, '0');

  if (hours > 0) return `${hours}h ${minutes}m ${remainder}s`;
  return `${minutes}m ${remainder}s`;
}

export function useFormattedJobRuntime(job?: Partial<JobResponse> | null) {
  const isRunning = job?.status === 'queued' || job?.status === 'running';
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    if (!isRunning) return undefined;
    setNow(Date.now());
    const interval = window.setInterval(() => setNow(Date.now()), 100);
    return () => window.clearInterval(interval);
  }, [isRunning, job?.job_id]);

  const elapsedSeconds = useMemo(() => {
    const start = parseTimestamp(job?.created_at) ?? parseTimestamp(job?.started_at);
    const finish = parseTimestamp(job?.finished_at);
    if (start != null) return ((finish ?? now) - start) / 1000;

    const backendElapsed = Number(job?.elapsed_seconds);
    return Number.isFinite(backendElapsed) ? backendElapsed : null;
  }, [job?.started_at, job?.created_at, job?.finished_at, job?.elapsed_seconds, now]);

  return job && elapsedSeconds != null ? formatJobDuration(elapsedSeconds) : null;
}

export default function JobRuntime({ job, label }: JobRuntimeProps) {
  const isRunning = job?.status === 'queued' || job?.status === 'running';
  const formattedRuntime = useFormattedJobRuntime(job);

  if (!job || !formattedRuntime) return null;

  const displayLabel = isRunning ? 'Elapsed time' : label;

  return (
    <div className="job-runtime" aria-label={`${label || 'Runtime'} ${formattedRuntime}`}>
      <Clock3 size={15} />
      {displayLabel ? <span>{displayLabel}</span> : null}
      <strong>{formattedRuntime}</strong>
    </div>
  );
}
