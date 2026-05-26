type ProgressBarProps = {
  progress?: number;
  message?: string;
  className?: string;
};

export default function ProgressBar({ progress = 0, message, className = '' }: ProgressBarProps) {
  return (
    <div className={`progress-wrap ${className}`.trim()}>
      <div className="progress-bar">
        <div className="progress-fill" style={{ width: `${progress}%` }} />
      </div>
      {message ? <div className="muted small">{message}</div> : null}
    </div>
  );
}
