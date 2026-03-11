import { NextResponse } from 'next/server';
import { prisma } from '@/lib/prisma'; 

export async function GET() {
  try {
    const latestTruck = await prisma.truckDetection.findFirst({
      orderBy: { createdAt: 'desc' },
    });

    return NextResponse.json(latestTruck || { message: "No trucks found" });
  } catch (error) {
    return NextResponse.json({ error: "Failed to fetch" }, { status: 500 });
  }
}
