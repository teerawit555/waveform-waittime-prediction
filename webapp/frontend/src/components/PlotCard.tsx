type PlotCardProps = {
  title: string;
  imageUrl?: string | null;
};

export default function PlotCard({ title, imageUrl }: PlotCardProps) {
  return (
    <div className="plot-card">
      <div className="plot-title">{title}</div>
      {imageUrl
        ? <div className="plot-image-wrap">
            <img src={imageUrl} alt={title} />
          </div>
        : <div className="plot-empty">Plot output will appear here after the job is completed.</div>
      }
    </div>
  );
}