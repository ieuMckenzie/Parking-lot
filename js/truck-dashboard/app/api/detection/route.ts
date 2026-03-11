import { NextResponse } from 'next/server';
import { prisma } from '@/lib/prisma';


export async function POST(request: Request) {
  try {
    const body = await request.json();
    console.log("Received from Python:", body);
    const newEntry = await prisma.truckDetection.create({
      data: {
        plateNumber: body.plate || "UNKNOWN",
        usdotNumber: body.usdot || "UNKNOWN",
        unitId: body.unit_id || "UNKNOWN",
      }
    });

    return NextResponse.json({ 
      status: 'success', 
      dbId: newEntry.id 
    });
  } catch (error) {
    console.error("Database Error:", error);
    return NextResponse.json(
      { error: "Failed to save to database", details: error }, 
      { status: 500 }
    );
  }
}

export async function GET() {
  return NextResponse.json({ message: "API is alive and waiting for POST requests." });
}