import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs/promises';

export async function GET(req: NextRequest) {
  const filePath = req.nextUrl.searchParams.get('path');

  if (!filePath) {
    return NextResponse.json({ error: 'Invalid file path' }, { status: 400 });
  }

  try {
    const fileContent = await fs.readFile(filePath, 'utf-8');
    return new NextResponse(fileContent, { status: 200 });
  } catch (error) {
    console.error('Error reading file:', error);
    return NextResponse.json({ error: 'Failed to read file' }, { status: 500 });
  }
}