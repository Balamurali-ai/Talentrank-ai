import "./globals.css";
import { TalentRankProvider } from "./context/TalentRankContext";

export const metadata = {
  title: "TalentRank AI - AI Recruitment Analytics",
  description: "AI-assisted resume screening and candidate ranking system",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>
        <TalentRankProvider>
          {children}
        </TalentRankProvider>
      </body>
    </html>
  );
}
