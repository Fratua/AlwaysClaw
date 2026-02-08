/**
 * OpenClawAgent - Winston Logger Configuration
 * Structured logging with file rotation for Windows service
 */

const winston = require('winston');
const path = require('path');
const fs = require('fs');

// Try to load winston-daily-rotate-file, fallback if not available
let DailyRotateFile;
try {
  DailyRotateFile = require('winston-daily-rotate-file');
} catch (e) {
  // Fallback to regular file transport
  DailyRotateFile = null;
}

const logDir = path.join(__dirname, 'logs');

// Ensure logs directory exists
if (!fs.existsSync(logDir)) {
  try {
    fs.mkdirSync(logDir, { recursive: true });
  } catch (error) {
    console.error('Failed to create logs directory:', error.message);
  }
}

// Define log levels (npm levels)
const levels = {
  error: 0,
  warn: 1,
  info: 2,
  http: 3,
  verbose: 4,
  debug: 5,
  silly: 6
};

// Define colors for each level
const colors = {
  error: 'red',
  warn: 'yellow',
  info: 'green',
  http: 'magenta',
  verbose: 'cyan',
  debug: 'blue',
  silly: 'gray'
};

winston.addColors(colors);

// Create transports array
const transports = [
  // Console transport for development
  new winston.transports.Console({
    level: process.env.NODE_ENV === 'production' ? 'warn' : 'debug',
    format: winston.format.combine(
      winston.format.colorize({ all: true }),
      winston.format.timestamp({ format: 'YYYY-MM-DD HH:mm:ss' }),
      winston.format.printf(({ level, message, timestamp, ...metadata }) => {
        let msg = `${timestamp} [${level}]: ${message}`;
        if (Object.keys(metadata).length > 0) {
          msg += ` ${JSON.stringify(metadata)}`;
        }
        return msg;
      })
    )
  })
];

// Add file transports if DailyRotateFile is available
if (DailyRotateFile) {
  transports.push(
    // Rotating file transport for all logs
    new DailyRotateFile({
      filename: path.join(logDir, 'application-%DATE%.log'),
      datePattern: 'YYYY-MM-DD',
      zippedArchive: true,
      maxSize: '20m',
      maxFiles: '14d',
      level: 'info',
      format: winston.format.combine(
        winston.format.timestamp({ format: 'YYYY-MM-DD HH:mm:ss' }),
        winston.format.errors({ stack: true }),
        winston.format.json()
      )
    }),
    
    // Separate file for error logs
    new DailyRotateFile({
      filename: path.join(logDir, 'error-%DATE%.log'),
      datePattern: 'YYYY-MM-DD',
      level: 'error',
      zippedArchive: true,
      maxSize: '20m',
      maxFiles: '30d',
      format: winston.format.combine(
        winston.format.timestamp({ format: 'YYYY-MM-DD HH:mm:ss' }),
        winston.format.errors({ stack: true }),
        winston.format.json()
      )
    })
  );
} else {
  // Fallback to regular file transport
  transports.push(
    new winston.transports.File({
      filename: path.join(logDir, 'application.log'),
      level: 'info',
      format: winston.format.combine(
        winston.format.timestamp({ format: 'YYYY-MM-DD HH:mm:ss' }),
        winston.format.errors({ stack: true }),
        winston.format.json()
      )
    }),
    new winston.transports.File({
      filename: path.join(logDir, 'error.log'),
      level: 'error',
      format: winston.format.combine(
        winston.format.timestamp({ format: 'YYYY-MM-DD HH:mm:ss' }),
        winston.format.errors({ stack: true }),
        winston.format.json()
      )
    })
  );
}

// Create logger instance
const logger = winston.createLogger({
  level: process.env.LOG_LEVEL || 'info',
  levels,
  format: winston.format.combine(
    winston.format.timestamp({ format: 'YYYY-MM-DD HH:mm:ss' }),
    winston.format.errors({ stack: true }),
    winston.format.splat(),
    winston.format.json()
  ),
  defaultMeta: {
    service: 'OpenClawAgent',
    pid: process.pid,
    workerType: process.env.WORKER_TYPE || 'master',
    workerId: process.env.WORKER_ID || 'master'
  },
  transports,
  exitOnError: false
});

// Create a stream object for Morgan HTTP logging
logger.stream = {
  write: (message) => {
    logger.http(message.trim());
  }
};

// Override console methods in production
if (process.env.NODE_ENV === 'production') {
  console.log = (...args) => logger.info.call(logger, args.join(' '));
  console.info = (...args) => logger.info.call(logger, args.join(' '));
  console.warn = (...args) => logger.warn.call(logger, args.join(' '));
  console.error = (...args) => logger.error.call(logger, args.join(' '));
  console.debug = (...args) => logger.debug.call(logger, args.join(' '));
}

module.exports = logger;
