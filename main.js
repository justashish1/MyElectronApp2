const { app, BrowserWindow } = require('electron');
const path = require('path');
const { exec } = require('child_process');
const fs = require('fs');
const winston = require('winston');

// Create a log directory if it does not exist
const logDir = 'log';
if (!fs.existsSync(logDir)) {
  fs.mkdirSync(logDir);
}

// Configure winston
const logger = winston.createLogger({
  level: 'info',
  format: winston.format.combine(
    winston.format.timestamp({
      format: 'YYYY-MM-DD HH:mm:ss'
    }),
    winston.format.printf(info => `${info.timestamp} ${info.level}: ${info.message}`)
  ),
  transports: [
    new winston.transports.File({ filename: path.join(logDir, 'app.log') })
  ]
});

function createWindow() {
  const mainWindow = new BrowserWindow({
    width: 1600,
    height: 900,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
    },
  });

  mainWindow.loadURL('http://localhost:8501');

  // Open the DevTools.
  // mainWindow.webContents.openDevTools();
}

app.on('ready', () => {
  logger.info('App is ready.');

  // Start the Streamlit app
  const streamlitProcess = exec('streamlit run STARENGTSANALYTICSAPP.py', (err, stdout, stderr) => {
    if (err) {
      logger.error(`Error starting Streamlit: ${err}`);
      return;
    }
  });

  streamlitProcess.stdout.on('data', (data) => {
    logger.info(`Streamlit stdout: ${data}`);
  });

  streamlitProcess.stderr.on('data', (data) => {
    logger.error(`Streamlit stderr: ${data}`);
  });

  streamlitProcess.on('close', (code) => {
    logger.info(`Streamlit process exited with code ${code}`);
  });

  createWindow();
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});
