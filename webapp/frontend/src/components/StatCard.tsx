// type StatCardProps = {
//   title: string;
//   value: string | number;
//   hint?: string;
// };

// function StatCard({ title, value, hint }: StatCardProps) {
//   return (
//     <div className="stat-card">
//       <div className="stat-title">{title}</div>
//       <div className="stat-value">{value}</div>
//       {hint ? <div className="stat-hint">{hint}</div> : null}
//     </div>
//   );
// }

// export default StatCard;

interface StatCardProps {
  title: string;
  value: string | number;
}

export default function StatCard({ title, value }: StatCardProps) {
  return (
    <div className="stat-card">
      <div className="stat-label">{title}</div>
      <div className="stat-value">{value}</div>
    </div>
  );
}
