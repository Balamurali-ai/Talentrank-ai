"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useTalentRank } from "./context/TalentRankContext";
import LoadingState from "./components/LoadingState";

export default function RootPage() {
  const router = useRouter();
  const { token, authChecked } = useTalentRank();

  useEffect(() => {
    if (!authChecked) return;
    if (token) {
      router.replace("/dashboard");
    } else {
      router.replace("/login");
    }
  }, [authChecked, token, router]);

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
      <LoadingState message="Loading TalentRank AI..." rows={2} />
    </div>
  );
}
