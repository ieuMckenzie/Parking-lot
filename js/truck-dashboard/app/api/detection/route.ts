import { PrismaClient } from '@prisma/client'
const prisma = new PrismaClient()

export async function POST(request: Request) {
  const { plate, usdot, unit_id } = await request.json();

  await prisma.truck_detections.create({
    data: { 
      plate_number: plate, 
      usdot_number: usdot, 
      unit_id: unit_id 
    }
  });

  const match = await prisma.dock_assignments.findUnique({
    where: { plate_number: plate }
  });

  const instructions = match
    ? `PROCEED TO ${match.assigned_dock}` 
    : "NO ASSIGNMENT FOUND - GO TO STAGING";

  return Response.json({ instructions });
}
