import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useToast } from "../hooks/use-toast"; // Keep your hook

const Login: React.FC = () => {
  const navigate = useNavigate();
  const { toast } = useToast();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const handleLogin = (e: React.FormEvent) => {
    e.preventDefault();
    if (email && password) {
      toast({ title: "Login successful", description: "Welcome back!" });
      navigate("/");
    } else {
      toast({ title: "Login failed", description: "Enter valid credentials" });
    }
  };

  return (
    <div style={{ minHeight: "100vh", display: "flex", alignItems: "center", justifyContent: "center", padding: "2rem", background: "#f5f5f5" }}>
      <div style={{ background: "#fff", padding: "2rem", borderRadius: "8px", width: "100%", maxWidth: "400px", boxShadow: "0 2px 10px rgba(0,0,0,0.1)" }}>
        <h2 style={{ textAlign: "center", marginBottom: "1rem" }}>Digital Twin Management</h2>
        <p style={{ textAlign: "center", marginBottom: "2rem" }}>Sign in to access your dashboard</p>
        <form onSubmit={handleLogin} style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
          <div style={{ display: "flex", flexDirection: "column" }}>
            <label>Email</label>
            <input
              type="email"
              placeholder="name@example.com"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              style={{ padding: "0.5rem", borderRadius: "4px", border: "1px solid #ccc" }}
            />
          </div>
          <div style={{ display: "flex", flexDirection: "column" }}>
            <label>Password</label>
            <input
              type="password"
              placeholder="Enter your password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              style={{ padding: "0.5rem", borderRadius: "4px", border: "1px solid #ccc" }}
            />
          </div>
          <button type="submit" style={{ padding: "0.75rem", borderRadius: "4px", background: "#6366f1", color: "#fff", fontWeight: "bold", cursor: "pointer" }}>
            Sign In
          </button>
          <div style={{ textAlign: "center", marginTop: "0.5rem" }}>
            <a href="#" style={{ fontSize: "0.875rem", color: "#6366f1" }}>Forgot password?</a>
          </div>
        </form>
      </div>
    </div>
  );
};

export default Login;
