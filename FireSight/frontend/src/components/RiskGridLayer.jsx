import { useEffect } from "react";
import { useMap } from "react-leaflet";
import L from "leaflet";

const CELL_SIZE = 0.125;
const HALF = CELL_SIZE / 2;

// Maps risk to color — tuned for this model's 0-0.5 range
function riskToStyle(risk) {
  if (risk < 0.03) return { color: "rgb(0,100,0)", opacity: 0.15 };
  if (risk < 0.08) return { color: "rgb(0,180,0)", opacity: 0.2 };

  let r, g;
  if (risk < 0.15) {
    // green to yellow-green
    const t = (risk - 0.08) / 0.07;
    r = Math.round(t * 200);
    g = 255;
    return { color: `rgb(${r},${g},0)`, opacity: 0.4 };
  }
  if (risk < 0.25) {
    // yellow to orange
    const t = (risk - 0.15) / 0.1;
    r = 255;
    g = Math.round(255 - t * 100);
    return { color: `rgb(${r},${g},0)`, opacity: 0.5 };
  }

  // orange to red (0.25+)
  const t = Math.min((risk - 0.25) / 0.25, 1);
  r = 255;
  g = Math.round(155 - t * 155);
  return { color: `rgb(${r},${g},0)`, opacity: 0.55 };
}

export default function RiskGridLayer({ riskGrid }) {
  const map = useMap();

  useEffect(() => {
    if (!riskGrid.length) return;

    const group = L.layerGroup();

    for (const cell of riskGrid) {
      const { color, opacity } = riskToStyle(cell.risk);
      L.rectangle(
        [[cell.lat - HALF, cell.lon - HALF], [cell.lat + HALF, cell.lon + HALF]],
        { fillColor: color, fillOpacity: opacity, weight: 0, interactive: false }
      ).addTo(group);
    }

    group.addTo(map);
    return () => map.removeLayer(group);
  }, [map, riskGrid]);

  return null;
}
