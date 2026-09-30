import "./globals.css";
import { AuthProvider } from "@/lib/auth";

export const metadata = {
  title: "DepthWizard — Single-View Satellite Height Estimation",
  description:
    "SIH26175: turn a single RGB satellite/aerial image into a navigable 3D elevation model.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>
        <AuthProvider>{children}</AuthProvider>
      </body>
    </html>
  );
}
