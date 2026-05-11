import http from 'http';
import dotenv from 'dotenv';
import app from './app.js';
import { connectDB, disconnectDB } from './config/db.js';

dotenv.config();

const PORT = process.env.PORT || 5000;

connectDB().then((dbInfo) => {
  const server = http.createServer(app);

  server.on('error', async (error) => {
    if (error.code === 'EADDRINUSE') {
      console.error(`Port ${PORT} is already in use. The backend is likely already running.`);
      await disconnectDB();
      process.exit(1);
    }

    console.error('Server failed to start:', error.message);
    await disconnectDB();
    process.exit(1);
  });

  server.listen(PORT, () => {
    console.log(`Server listening on port ${PORT} (${dbInfo.mode} database)`);
  });

  const shutdown = async () => {
    server.close(async () => {
      await disconnectDB();
      process.exit(0);
    });
  };

  process.on('SIGINT', shutdown);
  process.on('SIGTERM', shutdown);
});
