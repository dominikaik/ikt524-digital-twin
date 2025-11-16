// src/pages/Dashboard.tsx
import { useState } from "react";
import { RefreshCw, AlertCircle } from "lucide-react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";
import useForecast from "../hooks/useForecast";
import { useFiles } from "../hooks/useFiles";
import "../index.css";

// Reusable Button
interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  children: React.ReactNode;
  variant?: "default" | "outline" | "destructive";
}

const Button: React.FC<ButtonProps> = ({ children, variant = "default", ...props }) => (
  <button className={`btn ${variant}`} {...props}>{children}</button>
);

// Reusable Input
interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string;
}

const Input: React.FC<InputProps> = ({ label, ...props }) => (
  <div className="input-wrapper">
    {label && <label className="input-label">{label}</label>}
    <input className="input-field" {...props} />
  </div>
);

// Reusable Textarea
interface TextareaProps extends React.TextareaHTMLAttributes<HTMLTextAreaElement> {
  label?: string;
}

const Textarea: React.FC<TextareaProps> = ({ label, ...props }) => (
  <div className="textarea-wrapper">
    {label && <label className="textarea-label">{label}</label>}
    <textarea className="textarea-field" {...props} />
  </div>
);

// Types
type GraphView = "glucose" | "food" | "exercise" | "sleep" | "all";
type ViewMode = "input" | "results";

const Dashboard = () => {
  const forecast = useForecast();
  const fileManager = useFiles();

  const [graphView, setGraphView] = useState<GraphView>("all");
  const [viewMode, setViewMode] = useState<ViewMode>("input");

  // Form inputs
  const [currentGlucose, setCurrentGlucose] = useState("");
  const [insulin, setInsulin] = useState("");
  const [food, setFood] = useState("");
  const [exercise, setExercise] = useState(true);
  const [sleep, setSleep] = useState(true);
  const [useFuture, setUseFuture] = useState(false);

  // Handle form submission
  const handleSubmit = async () => {
    if (!fileManager.selectedFile) {
      alert("Please select a data file");
      return;
    }

    try {
      await forecast.generateForecast({
        filename: fileManager.selectedFile,
        customInputs: {
          glucose: parseFloat(currentGlucose) || 0,
          bolus: parseFloat(insulin) || 0,
          meal: parseFloat(food) || 0,
          exercise: exercise ? 1 : 0,
          sleep: sleep,
        },
        useFuture,
      });

      setViewMode("results");
      alert("Forecast generated successfully!");
    } catch (error) {
      console.error("Forecast error:", error);
      alert("Failed to generate forecast. Please try again.");
    }
  };


  // Render chart lines using CSV structure
  const renderLines = () => {
    switch (graphView) {
      case "glucose":
        return (
          <>
            <Line type="monotone" dataKey="csv_no_future" stroke="#EF4444" strokeWidth={2} strokeDasharray="5 5" name="Baseline" />
            <Line type="monotone" dataKey="csv_with_future" stroke="#3B82F6" strokeWidth={2} strokeDasharray="3 3" name="With Future" />
          </>
        );
      case "food":
        return <Line type="monotone" dataKey="d_t_bolus_carbs" stroke="#22c55e" strokeWidth={2} name="Meal + Insulin" />;
      case "exercise":
        return <Line type="monotone" dataKey="d_t_bolus_carbs_sleep_exercise" stroke="#06b6d4" strokeWidth={2} name="With Exercise" />;
      case "sleep":
        return <Line type="monotone" dataKey="d_t_bolus_carbs_sleep_exercise" stroke="#9333ea" strokeWidth={2} name="With Sleep" />;
      default:
        return (
          <>
            <Line type="monotone" dataKey="csv_no_future" stroke="#EF4444" strokeWidth={2} strokeDasharray="5 5" name="Baseline" />
            <Line type="monotone" dataKey="csv_with_future" stroke="#3B82F6" strokeWidth={2} strokeDasharray="3 3" name="With Future" />
            <Line type="monotone" dataKey="d_t_bolus_carbs" stroke="#22c55e" strokeWidth={2} name="Meal + Insulin" />
            <Line type="monotone" dataKey="d_t_bolus_carbs_sleep_exercise" stroke="#8B5CF6" strokeWidth={2} name="Sleep/Exercise" />
          </>
        );
    }
  };


  return (
    <div className="dashboard-container">
      <main className="dashboard-main">

        {/* Error Display */}
        {(forecast.error || fileManager.error) && (
          <div style={{
            backgroundColor: 'rgba(239, 68, 68, 0.1)',
            border: '1px solid #EF4444',
            borderRadius: '8px',
            padding: '16px',
            marginBottom: '16px',
            display: 'flex',
            alignItems: 'center',
            gap: '8px'
          }}>
            <AlertCircle style={{ width: '20px', height: '20px', color: '#EF4444' }} />
            <p style={{ color: '#EF4444', margin: 0 }}>
              {forecast.error || fileManager.error}
            </p>
          </div>
        )}

        {/* Chart */}
        <div className="chart-container">
          <ResponsiveContainer width="100%" height={350}>
            <LineChart data={forecast.chartData.filter(d => d.time >= 1)}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="time" type="number" domain={[1, 12]} ticks={[1,2,3,4,5,6,7,8,9,10,11,12]} label={{ value: "Time (min)", position: "insideBottom", offset: -5 }} />
              <YAxis label={{ value: "Glucose (mg/dL)", angle: -90, position: "insideLeft" }} />
              <Tooltip />
              <Legend />
              {renderLines()}
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Graph Buttons */}
        <div className="graph-buttons">
          {["glucose", "food", "exercise", "sleep", "all"].map((view) => (
            <Button
              key={view}
              variant={graphView === view ? "default" : "outline"}
              onClick={() => setGraphView(view as GraphView)}
            >
              {view.charAt(0).toUpperCase() + view.slice(1)}
            </Button>
          ))}
        </div>

        {/* View Mode Tabs */}
        <div className="view-mode-tabs">
          <Button variant={viewMode === "input" ? "default" : "outline"} onClick={() => setViewMode("input")}>
            Insert Information
          </Button>
          <Button variant={viewMode === "results" ? "default" : "outline"} onClick={() => setViewMode("results")}>
            Show Results
          </Button>
        </div>

        {/* Input Form */}
        {viewMode === "input" ? (
          <div className="input-form">
            <Input label="Current Glucose" type="number" value={currentGlucose} onChange={(e) => setCurrentGlucose(e.target.value)} />
            <Input label="Insulin (Units)" type="number" step="0.5" value={insulin} onChange={(e) => setInsulin(e.target.value)} />
            <Input label="Food (Carbs in grams)" type="number" step="5" value={food} onChange={(e) => setFood(e.target.value)} />

            <div className="toggle-group">
              <span>Exercise:</span>
              <Button variant={exercise ? "default" : "outline"} onClick={() => setExercise(true)}>Yes</Button>
              <Button variant={!exercise ? "default" : "outline"} onClick={() => setExercise(false)}>No</Button>
            </div>

            <div className="toggle-group">
              <span>Sleep:</span>
              <Button variant={sleep ? "default" : "outline"} onClick={() => setSleep(true)}>Yes</Button>
              <Button variant={!sleep ? "default" : "outline"} onClick={() => setSleep(false)}>No</Button>
            </div>

            <div className="toggle-group">
              <span>Use Future Data:</span>
              <Button variant={useFuture ? "default" : "outline"} onClick={() => setUseFuture(true)}>Yes</Button>
              <Button variant={!useFuture ? "default" : "outline"} onClick={() => setUseFuture(false)}>No</Button>
            </div>

            <Button onClick={handleSubmit} disabled={forecast.loading}>
              {forecast.loading ? (
                <>
                  <RefreshCw style={{ width: '16px', height: '16px', marginRight: '8px', animation: 'spin 1s linear infinite' }} />
                  Processing...
                </>
              ) : 'Generate Forecast'}
            </Button>
          </div>
        ) : (
          <div className="results-form">
            <Input label="Short Term Prediction" value={forecast.shortTermPrediction} readOnly />
            <Input label="Long Term Prediction" value={forecast.longTermPrediction} readOnly />
            <Textarea label="Intervention" value={forecast.interventions.map(i => `${i.scenario}: ${i.message}`).join("\n")} readOnly />
          </div>
        )}
      </main>
    </div>
  );
};

export default Dashboard;
