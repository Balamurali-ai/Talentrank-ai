"use client";

import { useState, useEffect } from "react";
import { useRouter, usePathname } from "next/navigation";
import { useTalentRank } from "../context/TalentRankContext";
import Sidebar from "./Sidebar";
import Topbar from "./Topbar";
import DisclaimerBanner from "./DisclaimerBanner";
import CandidateDetailDrawer from "./CandidateDetailDrawer";
import CandidateComparisonModal from "./CandidateComparisonModal";
import LoadingState from "./LoadingState";

export default function DashboardLayout({ children }) {
  const router = useRouter();
  const pathname = usePathname();
  const { token, authChecked } = useTalentRank();

  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  // Authentication Guard
  useEffect(() => {
    if (authChecked && !token) {
      router.push("/login");
    }
  }, [authChecked, token, router]);

  if (!authChecked || !token) {
    return (
      <div
        style={{
          minHeight: "100vh",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          backgroundColor: "var(--bg-canvas)",
        }}
      >
        <LoadingState message="Authenticating session..." rows={3} />
      </div>
    );
  }

  return (
    <div
      style={{
        display: "flex",
        minHeight: "100vh",
        backgroundColor: "var(--bg-canvas)",
      }}
    >
      {/* Sidebar */}
      <Sidebar
        collapsed={sidebarCollapsed}
        onToggleCollapse={() => setSidebarCollapsed(!sidebarCollapsed)}
        mobileOpen={mobileMenuOpen}
        onCloseMobile={() => setMobileMenuOpen(false)}
      />

      {/* Main Content Area */}
      <div
        style={{
          flex: 1,
          display: "flex",
          flexDirection: "column",
          minWidth: 0,
          overflowX: "hidden",
        }}
      >
        {/* Topbar */}
        <Topbar onOpenMobile={() => setMobileMenuOpen(true)} />

        {/* Global Academic Disclaimer */}
        <DisclaimerBanner />

        {/* Page Content Container */}
        <main
          style={{
            flex: 1,
            padding: "1.75rem 2rem",
            maxWidth: "1400px",
            width: "100%",
            margin: "0 auto",
          }}
        >
          {children}
        </main>
      </div>

      {/* Slide-over Detail Drawer */}
      <CandidateDetailDrawer />

      {/* Side-by-Side Comparison Modal */}
      <CandidateComparisonModal />
    </div>
  );
}
