import { readFile, writeFile, mkdir } from 'fs/promises'
import { join } from 'path'

// Define the path for the metadata file within the public/uploads directory
const UPLOADS_DIR = join(process.cwd(), 'public', 'uploads')
const METADATA_FILE_PATH = join(UPLOADS_DIR, 'files_metadata.json')

interface StoredFile {
  id: string // Unique ID (nanoid generated filename)
  originalName: string // Original filename
  url: string // Public URL to access the file (e.g., /api/files/original-name.jpg)
  uploadedAt: Date
}

interface FileMetadata {
  files: { [id: string]: StoredFile }
  nameToId: { [originalName: string]: string }
}

class FileStorage {
  private files = new Map<string, StoredFile>() // Map from unique ID to StoredFile
  private nameToId = new Map<string, string>() // Map from originalName to unique ID

  private constructor() {
    this._loadMetadata() // Load metadata when the singleton is instantiated
  }

  private static instance: FileStorage

  public static getInstance(): FileStorage {
    if (!FileStorage.instance) {
      FileStorage.instance = new FileStorage()
    }
    return FileStorage.instance
  }

  private async _loadMetadata(): Promise<void> {
    try {
      // Ensure the uploads directory exists before trying to read metadata
      await mkdir(UPLOADS_DIR, { recursive: true })

      const data = await readFile(METADATA_FILE_PATH, 'utf-8')
      const metadata: FileMetadata = JSON.parse(data)

      this.files = new Map(Object.entries(metadata.files).map(([id, file]) => [id, { ...file, uploadedAt: new Date(file.uploadedAt) }]))
      this.nameToId = new Map(Object.entries(metadata.nameToId))
      console.log('FileStorage: Metadata loaded successfully.')
    } catch (error: any) {
      if (error.code === 'ENOENT') {
        console.log('FileStorage: Metadata file not found, starting with empty storage.')
      } else {
        console.error('FileStorage: Error loading metadata:', error)
      }
    }
  }

  private async _saveMetadata(): Promise<void> {
    try {
      const metadata: FileMetadata = {
        files: Object.fromEntries(this.files),
        nameToId: Object.fromEntries(this.nameToId)
      }
      await writeFile(METADATA_FILE_PATH, JSON.stringify(metadata, null, 2), 'utf-8')
      console.log('FileStorage: Metadata saved successfully.')
    } catch (error) {
      console.error('FileStorage: Error saving metadata:', error)
    }
  }

  async storeFile(id: string, originalName: string, url: string): Promise<void> {
    const file: StoredFile = {
      id,
      originalName,
      url,
      uploadedAt: new Date()
    }

    this.files.set(id, file)
    this.nameToId.set(originalName, id) // Store original name -> ID mapping
    await this._saveMetadata()
  }

  getFileById(id: string): StoredFile | undefined {
    return this.files.get(id)
  }

  getFileByName(originalName: string): StoredFile | undefined {
    const id = this.nameToId.get(originalName)
    return id ? this.files.get(id) : undefined
  }

  getAllFiles(): StoredFile[] {
    return Array.from(this.files.values())
  }

  deleteFile(id: string): boolean {
    const file = this.files.get(id)
    if (file) {
      this.files.delete(id)
      this.nameToId.delete(file.originalName)
      return true
    }
    return false
  }
}

export const fileStorage = FileStorage.getInstance()
