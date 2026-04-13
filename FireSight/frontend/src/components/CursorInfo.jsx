import { useState, useMemo } from "react";
import { useMapEvents } from "react-leaflet";

const CELL_SIZE = 0.125;

export default function CursorInfo({ riskGrid }) {
  const [info, setInfo] = useState(null);

  // Build a fast lookup: "lat,lon" -> risk
  const riskMap = useMemo(() => {
    const m = new Map();
    for (const p of riskGrid) {
      // Use consistent key format matching the data
      const key = `${Number(p.lat).toFixed(4)},${Number(p.lon).toFixed(4)}`;
      m.set(key, p.risk);
    }
    return m;
  }, [riskGrid]);

  useMapEvents({
    mousemove(e) {
      const { lat, lng } = e.latlng;
      const snappedLat = (Math.round(lat / CELL_SIZE) * CELL_SIZE).toFixed(4);
      const snappedLon = (Math.round(lng / CELL_SIZE) * CELL_SIZE).toFixed(4);
      const risk = riskMap.get(`${snappedLat},${snappedLon}`);

      setInfo({
        lat: lat.toFixed(4),
        lon: lng.toFixed(4),
        risk: risk != null ? (risk * 100).toFixed(1) : null,
      });
    },
    mouseout() { setInfo(null); },
  });

  if (!info) return null;

  const riskColor = info.risk == null ? "#888"
    : info.risk > 25 ? "#ff4400"
    : info.risk > 15 ? "#ff8800"
    : info.risk > 8 ? "#ffff00"
    : "#00ff00";

  return (
    <div className="cursor-info">
      <div>{info.lat}, {info.lon}</div>
      {info.risk != null
        ? <div style={{ color: riskColor, fontWeight: 700 }}>Fire Risk: {info.risk}%</div>
        : <div style={{ color: "#888" }}>No data</div>
      }
    </div>
  );
}
