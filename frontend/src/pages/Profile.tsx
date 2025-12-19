import React from "react";
import { useNavigate } from "react-router-dom";

const Profile: React.FC = () => {
  const navigate = useNavigate();

  const userInfo = {
    name: "Dr. Sarah Johnson",
    email: "sarah.johnson@example.com",
    role: "Patient",
    joinDate: "January 2024",
    lastLogin: "Today at 9:45 AM",
  };

  const infoBoxStyle: React.CSSProperties = {
    display: "flex",
    gap: "0.5rem",
    padding: "0.75rem",
    borderRadius: "6px",
    background: "#eee",
  };

  return (
    <div style={{ minHeight: "100vh", background: "#f5f5f5", padding: "2rem" }}>
      <header style={{ display: "flex", justifyContent: "flex-end", gap: "1rem" }}>
        <button onClick={() => navigate("/profile")}>Profile</button>
      </header>

      <main style={{ maxWidth: "800px", margin: "2rem auto" }}>
        <div style={{ background: "#fff", padding: "2rem", borderRadius: "8px", boxShadow: "0 2px 10px rgba(0,0,0,0.1)" }}>
          <div style={{ textAlign: "center", marginBottom: "2rem" }}>
            <div style={{ margin: "0 auto 1rem", height: "100px", width: "100px", borderRadius: "50%", background: "#ddd", display: "flex", alignItems: "center", justifyContent: "center" }}>
              {/* You can use an icon here */}
              <span>👤</span>
            </div>
            <h2>{userInfo.name}</h2>
            <p>{userInfo.role}</p>
          </div>

          <div style={{ marginBottom: "1rem" }}>
            <h3>Account Information</h3>
            <div style={infoBoxStyle}>
              <strong>Email:</strong>
              <span>{userInfo.email}</span>
            </div>
            <div style={infoBoxStyle}>
              <strong>Member Since:</strong>
              <span>{userInfo.joinDate}</span>
            </div>
            <div style={infoBoxStyle}>
              <strong>Last Login:</strong>
              <span>{userInfo.lastLogin}</span>
            </div>
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
            <button style={{ padding: "0.75rem", borderRadius: "4px", border: "1px solid #6366f1", background: "#fff", color: "#6366f1" }}>Edit Profile</button>
            <button style={{ padding: "0.75rem", borderRadius: "4px", border: "1px solid #6366f1", background: "#fff", color: "#6366f1" }}>Change Password</button>
            <button style={{ padding: "0.75rem", borderRadius: "4px", border: "1px solid #f87171", background: "#f87171", color: "#fff" }} onClick={() => navigate("/login")}>
              Sign Out
            </button>
          </div>

          <div style={{ marginTop: "1rem" }}>
            <button style={{ padding: "0.75rem", borderRadius: "4px", width: "100%" }} onClick={() => navigate("/")}>
              Back to Dashboard
            </button>
          </div>
        </div>
      </main>
    </div>
  );
};

export default Profile;
