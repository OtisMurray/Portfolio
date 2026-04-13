export default function Legend() {
  return (
    <div className="legend">
      <h4>Wildfire Risk</h4>
      <div className="legend-gradient" />
      <div className="legend-labels">
        <span>Low</span>
        <span>Medium</span>
        <span>High</span>
      </div>
      <div className="legend-item">
        <div className="legend-dot" style={{ background: "#ff3333", borderColor: "#ff0000" }} />
        <span>NASA FIRMS live fire (24h)</span>
      </div>
    </div>
  );
}
