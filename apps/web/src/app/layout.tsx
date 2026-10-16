import type { Metadata } from "next";
import { Inter_Tight, IBM_Plex_Mono } from "next/font/google";
import "./globals.css";
import { AppStateProvider } from "@/components/pg/app-state";
import { Header } from "@/components/pg/header";
import { TooltipProvider } from "@/components/ui/tooltip";

const inter = Inter_Tight({ variable: "--font-inter-tight", subsets: ["latin"], weight: ["400", "500", "600", "700"] });
const plex = IBM_Plex_Mono({ variable: "--font-plex-mono", subsets: ["latin"], weight: ["400", "500"] });

export const metadata: Metadata = {
  title: "Pragmatic Gym",
  description: "A research workbench for teaching language models social intelligence through interaction.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body className={`${inter.variable} ${plex.variable} antialiased`}>
        <AppStateProvider>
          <TooltipProvider delayDuration={150}>
            <Header />
            <main className="min-h-[calc(100vh-140px)]">{children}</main>
            <footer className="gutter mt-24 pb-10">
              <div className="rule-heavy mb-4" />
              <div className="flex flex-wrap items-start justify-between gap-4 t-small text-ink-2">
                <p className="max-w-[720px]">
                  Pragmatic Gym is an independent research prototype. All scenarios are synthetic.
                </p>
                <p className="font-bold text-ink">Diagnose → Intervene → Generate → Train → Evaluate</p>
              </div>
            </footer>
          </TooltipProvider>
        </AppStateProvider>
      </body>
    </html>
  );
}
