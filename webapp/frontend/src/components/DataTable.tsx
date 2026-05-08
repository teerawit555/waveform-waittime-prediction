type DataRow = Record<string, unknown>;

type DataTableProps = {
  rows: DataRow[];
};

function formatCell(value: unknown) {
  if (value === null || value === undefined || value === '') return '—';
  if (typeof value === 'number') return Number.isFinite(value) ? value.toString() : '—';
  return String(value);
}

function DataTable({ rows }: DataTableProps) {
  if (!rows || rows.length === 0) {
    return <div className="table-empty">No rows to display.</div>;
  }

  const columns = Object.keys(rows[0]);

  return (
    <div className="table-shell">
      <div className="table-scroll">
        <table className="data-table">
          <thead>
            <tr>
              {columns.map((col) => (
                <th key={col}>{col}</th>
              ))}
            </tr>
          </thead>

          <tbody>
            {rows.map((row, rowIndex) => (
              <tr key={rowIndex}>
                {columns.map((col) => (
                  <td key={`${rowIndex}-${col}`}>{formatCell(row[col])}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default DataTable;