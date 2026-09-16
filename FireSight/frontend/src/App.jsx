import { useState, useEffect } from "react";
import axios from "axios";
import FireMap from "./components/FireMap";
import Legend from "./components/Legend";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
const STATIC_RISK_GRID_URL = "/predictions.json";

export default function App() {
  const [riskGrid, setRiskGrid] = useState([]);
  const [liveFires, setLiveFires] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function fetchData() {
      try {
        let grid = [];
        try {
          const gridRes = await axios.get(`${API_URL}/api/risk-grid`);
          grid = gridRes.data;
        } catch {
          const fallbackGridRes = await axios.get(STATIC_RISK_GRID_URL);
          grid = fallbackGridRes.data;
        }

        let fires = [];
        try {
          const firesRes = await axios.get(`${API_URL}/api/live-fires`);
          fires = firesRes.data;
        } catch {
          fires = [];
        }

        setRiskGrid(Array.isArray(grid) ? grid : []);
        setLiveFires(Array.isArray(fires) ? fires : []);
      } catch (err) {
        setError("Forecast data is temporarily unavailable.");
      } finally {
        setLoading(false);
      }
    }
    fetchData();
  }, []);

  return (
    <div className="app">
      <header className="header">
        <h1><span>Fire</span>Sight</h1>
        <div className="header-stats">
          {riskGrid.length > 0 && `${riskGrid.length} grid cells`}
          {riskGrid.length > 0 && liveFires.length > 0 && " | "}
          {liveFires.length > 0 && `${liveFires.length} active fires`}
        </div>
      </header>

      {error && <div className="error-banner">{error}</div>}

      <div className="map-container">
        {loading && (
          <div className="loading-overlay">
            <div className="spinner" />
            <p>Loading wildfire data...</p>
          </div>
        )}
        <FireMap riskGrid={riskGrid} liveFires={liveFires} />
        <Legend />
      </div>
    </div>
  );
}
