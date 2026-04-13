import { MapContainer, TileLayer, CircleMarker, Popup } from "react-leaflet";
import RiskGridLayer from "./RiskGridLayer";
import CursorInfo from "./CursorInfo";

const CA_CENTER = [37, -119];
const CA_ZOOM = 6;

export default function FireMap({ riskGrid, liveFires }) {
  return (
    <MapContainer
      center={CA_CENTER}
      zoom={CA_ZOOM}
      style={{ width: "100%", height: "100%" }}
      zoomControl={true}
      preferCanvas={true}
    >
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> | &copy; <a href="https://carto.com/">CARTO</a>'
        url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
      />

      <RiskGridLayer riskGrid={riskGrid} />

      {liveFires.map((fire, i) => (
        <CircleMarker
          key={i}
          center={[fire.lat, fire.lon]}
          radius={5}
          pathOptions={{
            color: "#ff0000",
            fillColor: "#ff3333",
            fillOpacity: 0.9,
            weight: 2,
          }}
        >
          <Popup>
            <strong>Active Fire Detection</strong>
            <br />
            Lat: {fire.lat.toFixed(3)}, Lon: {fire.lon.toFixed(3)}
            <br />
            Brightness: {fire.brightness.toFixed(1)} K
          </Popup>
        </CircleMarker>
      ))}

      <CursorInfo riskGrid={riskGrid} />
    </MapContainer>
  );
}
