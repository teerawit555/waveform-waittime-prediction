type PlotCardProps = {
  title: string;
  imageUrl?: string | null;
  eyebrow?: string;
  description?: string;
  variant?: 'standard' | 'hero' | 'compact';
};

export default function PlotCard({
  title,
  imageUrl,
  eyebrow,
  description,
  variant = 'standard',
}: PlotCardProps) {
  return (
    <div className={`plot-card plot-card-${variant}`}>
      <div className="plot-title">
        {eyebrow ? <span>{eyebrow}</span> : null}
        <strong>{title}</strong>
        {description ? <small>{description}</small> : null}
      </div>
      {imageUrl
        ? <div className="plot-image-wrap">
            <img src={imageUrl} alt={title} />
          </div>
        : <div className="plot-empty">Plot output will appear here after the job is completed.</div>
      }
    </div>
  );
}
