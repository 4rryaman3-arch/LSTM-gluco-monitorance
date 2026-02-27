import type { Metadata } from "next";
import { Space_Grotesk, Sora } from "next/font/google";
import "./globals.css";

const headline = Space_Grotesk({
  subsets: ["latin"],
  variable: "--font-headline"
});

const body = Sora({
  subsets: ["latin"],
  variable: "--font-body"
});

export const metadata: Metadata = {
  title: "Gluco Monitorance",
  description: "Realtime LSTM-based CGM projection with insulin scenario mapping"
};

export default function RootLayout({
  children
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body className={`${headline.variable} ${body.variable}`}>{children}</body>
    </html>
  );
}
