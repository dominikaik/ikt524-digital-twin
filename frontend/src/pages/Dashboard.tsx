import { useEffect, useState } from "react";
import { User } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, } from "recharts";
import { useToast } from "../hooks/use-toast";
import "../index.css";

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  children: React.ReactNode;
  variant?: "default" | "outline" | "destructive";
}

const Button: React.FC<ButtonProps> = ({
  children,
  variant = "default",
  ...props
}) => <button className={`btn ${variant}`} {...props}>{children}</button>;

interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string;
}

const Input: React.FC<InputProps> = ({ label, ...props }) => (
  <div className="input-wrapper">
    {label && <label className="input-label">{label}</label>}
    <input className="input-field" {...props} />
  </div>
);

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

type DigitalTwinData = {
  time_min: number[];
  baseline: number[];
  meal45: number[];
  meal45_bolus3: number[];
  exercise: number[];
  sleep: number[];
};

type ChartDataPoint = {
  time: number;
  baseline: number;
  meal45: number;
  meal45_bolus3: number;
  exercise: number;
  sleep: number;
};

// Dashboard Component

const Dashboard = () => {
  const navigate = useNavigate();
  const { toast } = useToast();

  const [graphView, setGraphView] = useState<GraphView>("all");
  const [viewMode, setViewMode] = useState<ViewMode>("input");
  const [chartData, setChartData] = useState<ChartDataPoint[]>([]);

  // Form states
  const [currentGlucose, setCurrentGlucose] = useState("");
  const [insulin, setInsulin] = useState("");
  const [food, setFood] = useState("");
  const [exercise, setExercise] = useState(true);
  const [sleep, setSleep] = useState(true);
  const [shortTermPrediction] = useState("");
  const [longTermPrediction] = useState("");
  const [intervention] = useState(
    "Description based on the results no input"
  );

  // Fetch data
  useEffect(() => {
    fetch("/digital_twin_output.json")
      .then((res) => res.json())
      .then((data: DigitalTwinData) => {
        const formatted: ChartDataPoint[] = data.time_min.map((t, i) => ({
          time: t,
          baseline: data.baseline[i],
          meal45: data.meal45[i],
          meal45_bolus3: data.meal45_bolus3[i],
          exercise: data.exercise[i],
          sleep: data.sleep[i],
        }));
        setChartData(formatted);
      })
      .catch((err) => console.error("Error loading data:", err));
  }, []);

  const handleSubmit = () => {
    toast({ title: "Form submitted!", description: "Data saved successfully." });
    console.log("Form submitted");
  };

  const renderLines = () => {
    switch (graphView) {
      case "glucose":
        return <Line type="monotone" dataKey="baseline" stroke="#6366f1" dot={false} strokeWidth={2} />;
      case "food":
        return (
          <>
            <Line type="monotone" dataKey="meal45" stroke="#f97316" dot={false} strokeWidth={2} />
            <Line type="monotone" dataKey="meal45_bolus3" stroke="#22c55e" dot={false} strokeWidth={2} />
          </>
        );
      case "exercise":
        return <Line type="monotone" dataKey="exercise" stroke="#06b6d4" dot={false} strokeWidth={2} />;
      case "sleep":
        return <Line type="monotone" dataKey="sleep" stroke="#9333ea" dot={false} strokeWidth={2} />;
      default:
        return (
          <>
            <Line type="monotone" dataKey="baseline" stroke="#6366f1" dot={false} strokeWidth={2} />
            <Line type="monotone" dataKey="meal45" stroke="#f97316" dot={false} strokeWidth={2} />
            <Line type="monotone" dataKey="meal45_bolus3" stroke="#22c55e" dot={false} strokeWidth={2} />
            <Line type="monotone" dataKey="exercise" stroke="#06b6d4" dot={false} strokeWidth={2} />
            <Line type="monotone" dataKey="sleep" stroke="#9333ea" dot={false} strokeWidth={2} />
          </>
        );
    }
  };

  return (
    <div className="dashboard-container">

      <main className="dashboard-main">

        <div className="chart-container">
          <ResponsiveContainer width="100%" height={350}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="time" label={{ value: "Time (min)", position: "insideBottom", offset: -5 }} />
              <YAxis label={{ value: "Glucose (mg/dL)", angle: -90, position: "insideLeft" }} />
              <Tooltip />
              <Legend />
              {renderLines()}
            </LineChart>
          </ResponsiveContainer>
        </div>

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


        <div className="view-mode-tabs">
          <Button variant={viewMode === "input" ? "default" : "outline"} onClick={() => setViewMode("input")}>
            Insert Information
          </Button>
          <Button variant={viewMode === "results" ? "default" : "outline"} onClick={() => setViewMode("results")}>
            Show Results
          </Button>
        </div>

        {viewMode === "input" ? (
          <div className="input-form">
            <Input label="Current Glucose" value={currentGlucose} onChange={(e) => setCurrentGlucose(e.target.value)} />
            <Input label="Insulin" value={insulin} onChange={(e) => setInsulin(e.target.value)} />
            <Input label="Food" value={food} onChange={(e) => setFood(e.target.value)} />

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

            <Button onClick={handleSubmit}>Submit</Button>
          </div>
        ) : (
          <div className="results-form">
            <Input label="Short Term Prediction" value={shortTermPrediction} readOnly />
            <Input label="Long Term Prediction" value={longTermPrediction} readOnly />
            <Textarea label="Intervention" value={intervention} readOnly />
          </div>
        )}
      </main>
    </div>
  );
};

export default Dashboard;
