type SectionDividerProps = {
  label: string;
};

export default function SectionDivider({ label }: SectionDividerProps) {
  return (
    <div className="section-divider">
      <span>{label}</span>
    </div>
  );
}
