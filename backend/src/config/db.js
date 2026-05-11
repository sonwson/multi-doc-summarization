import mongoose from 'mongoose';
import { MongoMemoryServer } from 'mongodb-memory-server';

let memoryServer;

export const connectDB = async () => {
  try {
    await mongoose.connect(process.env.MONGODB_URI);
    console.log('MongoDB connected');
    return { mode: 'external', uri: process.env.MONGODB_URI };
  } catch (error) {
    const allowInMemoryDb = process.env.ALLOW_IN_MEMORY_DB !== 'false';

    if (!allowInMemoryDb) {
      console.error('MongoDB connection failed:', error.message);
      process.exit(1);
    }

    console.warn(`MongoDB unavailable (${error.message}). Falling back to in-memory MongoDB.`);
    memoryServer = await MongoMemoryServer.create({
      instance: { dbName: 'multi-doc-summarization' }
    });

    const uri = memoryServer.getUri();
    await mongoose.connect(uri);
    console.log('In-memory MongoDB connected');
    return { mode: 'memory', uri };
  }
};

export const disconnectDB = async () => {
  await mongoose.connection.close();
  if (memoryServer) {
    await memoryServer.stop();
    memoryServer = undefined;
  }
};
