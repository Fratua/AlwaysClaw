declare module 'sqlite3' {
  export class Database {}
  export class Statement {}
  export function verbose(): typeof import('sqlite3');
}
