export default function Home() {
  return (
    <main className="p-10 bk-slate-900 min-h-screen text-white">
      <h1 className="text-3xl font-bold mb-6">Detection</h1>
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Placeholder for real-time data */}
        <div className="p-6 bg-slate-800 rounded-lg border border-slate-700">
          <h2 className="text-sm text-slate-400 uppercase">License Plate</h2>
          <p className="text-2xl font-mono mt-2">1234</p>
        </div>
        
        <div className="p-6 bg-slate-800 rounded-lg border border-slate-700">
          <h2 className="text-sm text-slate-400 uppercase">USDOT Number</h2>
          <p className="text-2xl font-mono mt-2">USDOT 1</p>
        </div>

        <div className="p-6 bg-slate-800 rounded-lg border border-slate-700">
          <h2 className="text-sm text-slate-400 uppercase">Container / Trailer ID</h2>
          <p className="text-2xl font-mono mt-2">4 letters, 7 digits</p>
        </div>
      </div>
    </main>
  );
}