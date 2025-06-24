"""
Simple content processor for creating chunks from scraped text files.
"""

import os
import json
import glob


class SimpleChunker:
    """Simple text chunker that creates manageable pieces of content."""
    
    def __init__(self, chunk_size=500, overlap=50):
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        # Setup directories
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.raw_dir = os.path.join(base_dir, "data", "raw")
        self.processed_dir = os.path.join(base_dir, "data", "processed")
        os.makedirs(self.processed_dir, exist_ok=True)
    
    def chunk_text(self, text, source_name):
        """Split text into overlapping chunks."""
        if not text or len(text) < 50:
            return []
        
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            chunks.append({
                'text': chunk_text,
                'source': source_name,
                'chunk_id': f"{source_name}_chunk_{len(chunks) + 1}",
                'word_count': len(chunk_words),
                'char_count': len(chunk_text)
            })
            
            # Break if we've reached the end
            if i + self.chunk_size >= len(words):
                break
        
        return chunks
    
    def process_all_files(self):
        """Process all text files in the raw directory."""
        print("Processing scraped content files...")
        
        all_chunks = []
        txt_files = glob.glob(os.path.join(self.raw_dir, "*.txt"))
        
        for file_path in txt_files:
            file_name = os.path.basename(file_path)
            source_name = file_name.replace('.txt', '')
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                
                if content:
                    chunks = self.chunk_text(content, source_name)
                    all_chunks.extend(chunks)
                    print(f"  ✓ {source_name}: {len(chunks)} chunks ({len(content)} chars)")
                else:
                    print(f"  ⚠ {source_name}: empty file")
                    
            except Exception as e:
                print(f"  ✗ {source_name}: error - {e}")
        
        # Save chunks to JSON
        if all_chunks:
            output_path = os.path.join(self.processed_dir, "chunks.json")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(all_chunks, f, ensure_ascii=False, indent=2)
            
            print(f"\n✓ Created {len(all_chunks)} chunks")
            print(f"✓ Saved to: {output_path}")
        else:
            print("\n⚠ No chunks created")
        
        return all_chunks


def main():
    """Process all scraped content."""
    chunker = SimpleChunker()
    chunker.process_all_files()


if __name__ == "__main__":
    main()
