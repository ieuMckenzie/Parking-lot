'use client'; 

import { useEffect, useState } from 'react';

export default function Home() {
  const [status, setStatus] = useState({ 
    plate: "----", 
    dock: "AWAITING DETECTION",
    unitId: "----" 
  });

  useEffect(() => {
    const pollDatabase = async () => {
      try {
        const res = await fetch('/api/get-latest-truck'); 
        const data = await res.json();
        
        if (data) {
          setStatus({ 
            plate: data.plateNumber || "----", 
            dock: data.dock || "STAGING",
            unitId: data.unitId || "----"
          });
        }
      } catch (err) {
        console.error("Connection to API failed", err);
      }
    };

    const interval = setInterval(pollDatabase, 2000);
    return () => clearInterval(interval); 
  }, []);

  return (
    <main className="p-10 bg-slate-900 min-h-screen text-white font-sans">
      {/* Big Instruction Header */}
      <div className={`p-12 rounded-2xl border-4 text-center transition-all duration-500 ${
        status.dock.includes("DOCK") ? 'border-green-500 bg-green-500/10' : 'border-slate-700 bg-slate-800'
      }`}>
        <h1 className="text-xl uppercase tracking-widest text-slate-400">Gate Instruction</h1>
        <p className="text-8xl font-black mt-4">{status.dock}</p>
      </div>

      {/* Data Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-8">
        <div className="p-8 bg-slate-800 rounded-xl border border-slate-700">
          <h2 className="text-sm text-slate-400 uppercase">Latest Plate</h2>
          <p className="text-4xl font-mono mt-2 text-white-400">{status.plate}</p>
        </div>
        
        <div className="p-8 bg-slate-800 rounded-xl border border-slate-700">
          <h2 className="text-sm text-slate-400 uppercase">Unit / Trailer ID</h2>
          <p className="text-4xl font-mono mt-2 text-white-400">{status.unitId}</p>
        </div>
      </div>
    </main>
  );
}