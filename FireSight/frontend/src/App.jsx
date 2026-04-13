import { useState, useEffect } from "react";
import axios from "axios";
import FireMap from "./components/FireMap";
import Legend from "./components/Legend";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export default function App() {
  const [riskGrid, setRiskGrid] = useState([]);
  const [liveFires, setLiveFires] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function fetchData() {
      try {
        const [gridRes, firesRes] = await Promise.allSettled([
          axios.get(`${API_URL}/api/risk-grid`),
          axios.get(`${API_URL}/api/live-fires`),
        ]);

        if (gridRes.status === "fulfilled") setRiskGrid(gridRes.value.data);
        if (firesRes.status === "fulfilled") setLiveFires(firesRes.value.data);

        if (gridRes.status === "rejected" && firesRes.status === "rejected") {
          setError("Failed to connect to API. Is the backend running?");
        }
      } catch (err) {
        setError("Unexpected error loading data");
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
