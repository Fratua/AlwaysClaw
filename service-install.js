/**
 * OpenClawAgent - Windows Service Installer
 * Installs, configures, and manages the Windows service
 */

const Service = require('node-windows').Service;
const path = require('path');

const svc = new Service({
  name: 'OpenClawAgent',
  description: 'OpenClaw AI Agent System - 24/7 Background Service with GPT-5.2 integration',
  script: path.join(__dirname, 'service.js'),
  nodeOptions: [
    '--harmony',
    '--max-old-space-size=4096',
    '--optimize-for-size'
  ],
  workingDirectory: __dirname,
  wait: 2,
  grow: 0.5,
  abortOnError: false,
  logMode: 'rotate',
  logpath: path.join(__dirname, 'logs')
});

// Service event handlers
svc.on('install', () => {
  console.log('╔════════════════════════════════════════════════════════════╗');
  console.log('║     OpenClawAgent Service Installed Successfully          ║');
  console.log('╚════════════════════════════════════════════════════════════╝');
  console.log('Starting service...');
  svc.start();
});

svc.on('alreadyinstalled', () => {
  console.log('Service is already installed.');
  console.log('Use "npm run service:restart" to restart the service.');
});

svc.on('invalidinstallation', () => {
  console.error('ERROR: Invalid service installation');
  process.exit(1);
});

svc.on('uninstall', () => {
  console.log('╔════════════════════════════════════════════════════════════╗');
  console.log('║     OpenClawAgent Service Uninstalled Successfully        ║');
  console.log('╚════════════════════════════════════════════════════════════╝');
});

svc.on('start', () => {
  console.log('╔════════════════════════════════════════════════════════════╗');
  console.log('║     OpenClawAgent Service Started                         ║');
  console.log('╠════════════════════════════════════════════════════════════╣');
  console.log('║  Status: Running                                          ║');
  console.log('║  Logs: ./logs/                                            ║');
  console.log('║  Config: ./config/                                        ║');
  console.log('╚════════════════════════════════════════════════════════════╝');
});

svc.on('stop', () => {
  console.log('OpenClawAgent service stopped.');
});

svc.on('error', (error) => {
  console.error('Service error:', error);
});

// Command line handling
const command = process.argv[2];

switch(command) {
  case 'install':
    console.log('Installing OpenClawAgent service...');
    svc.install();
    break;
  case 'uninstall':
    console.log('Uninstalling OpenClawAgent service...');
    svc.uninstall();
    break;
  case 'start':
    console.log('Starting OpenClawAgent service...');
    svc.start();
    break;
  case 'stop':
    console.log('Stopping OpenClawAgent service...');
    svc.stop();
    break;
  case 'restart':
    console.log('Restarting OpenClawAgent service...');
    svc.stop();
    setTimeout(() => svc.start(), 3000);
    break;
  case 'status':
    console.log('Checking OpenClawAgent service status...');
    if (svc.exists) {
      console.log('Service is installed');
      // Try to check if running
      const { execSync } = require('child_process');
      try {
        const output = execSync('sc query OpenClawAgent', { encoding: 'utf8' });
        console.log(output);
      } catch (e) {
        console.log('Could not query service status');
      }
    } else {
      console.log('Service is not installed');
    }
    break;
  default:
    console.log(`
╔════════════════════════════════════════════════════════════╗
║     OpenClawAgent Service Manager                          ║
╠════════════════════════════════════════════════════════════╣
║  Usage: node service-install.js [command]                  ║
║                                                            ║
║  Commands:                                                 ║
║    install      Install the Windows service                ║
║    uninstall    Remove the Windows service                 ║
║    start        Start the service                          ║
║    stop         Stop the service                           ║
║    restart      Restart the service                        ║
║    status       Check service status                      ║
║                                                            ║
║  Or use npm scripts:                                       ║
║    npm run service:install                                 ║
║    npm run service:uninstall                               ║
║    npm run service:start                                   ║
║    npm run service:stop                                    ║
║    npm run service:restart                                 ║
╚════════════════════════════════════════════════════════════╝
    `);
}

module.exports = svc;
