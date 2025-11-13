import json
import pandas as pd
import multiprocessing as mp
from pathlib import Path
from tqdm import tqdm
from functools import partial
import re
import csv
import tempfile
import shutil
import os

class AcademicDataPipeline:
    def __init__(self, dataset_type="combined"):
        """
        Args:
            dataset_type: "train", "test", or "combined" (default)
        """
        self.dataset_type = dataset_type
        self.processed_dir = f"data/processed/{dataset_type}" if dataset_type != "combined" else "data/processed"
        Path(self.processed_dir).mkdir(parents=True, exist_ok=True)
        
        # Fields to remove in step 1
        self.fields_to_remove = [
            "page_start", "page_end", "doc_type", "lang",
            "volume", "issue", "issn", "isbn", "doi", "url"
        ]
        
        # Stopwords for keyword extraction
        self.stopwords = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 
            'to', 'was', 'will', 'with', 'we', 'our', 'this', 'these', 'using',
            'based', 'can', 'via', 'new', 'approach', 'method', 'methods'
        }
    
    def clean_line(self, line):
        """Step 1: Remove unnecessary fields from a single line"""
        if not line.strip():
            return None
        
        try:
            data = json.loads(line)
            
            # Remove specified fields if present
            for field in self.fields_to_remove:
                if field in data:
                    del data[field]
            
            return data  # Return dict instead of JSON string
        except json.JSONDecodeError:
            return None
    
    def extract_keywords_from_title(self, title, max_keywords=5):
        """Extract keywords from title"""
        if not title:
            return []
        
        # Convert to lowercase and tokenize
        words = re.findall(r'\b[a-z]{3,}\b', title.lower())
        
        # Filter stopwords
        keywords = [w for w in words if w not in self.stopwords]
        
        # Return top N keywords
        return keywords[:max_keywords]
    
    def process_and_extract(self, line):
        """Step 1+2: Clean line and extract entities/relationships"""
        # Step 1: Clean the line
        paper = self.clean_line(line)
        if paper is None:
            return None
        
        # Step 2: Extract entities and relationships
        paper_id = str(paper.get("id", ""))
        if not paper_id:
            return None
        
        result = {
            'paper': None,
            'authors': [],
            'keywords': [],
            'written_by': [],
            'cites': [],
            'has_keyword': []
        }
        
        # Extract paper information
        result['paper'] = {
            "paper_id": paper_id,
            "title": str(paper.get("title", ""))[:500],
            "abstract": str(paper.get("abstract", ""))[:2000],
            "year": paper.get("year", ""),
            "venue": str(paper.get("venue", ""))[:200],
            "n_citation": int(paper.get("n_citation", 0))
        }
        
        # Extract authors and written_by relationships
        authors = paper.get("authors", [])
        if isinstance(authors, list):
            for idx, author in enumerate(authors):
                if isinstance(author, dict):
                    author_id = str(author.get("id", "")).strip()
                    author_name = str(author.get("name", "Unknown")).strip()
                    author_org = str(author.get("org", "")).strip()
                    
                    # Generate author_id if empty
                    if not author_id:
                        author_id = f"author_{hash(author_name) % 10000000}"
                    
                    result['authors'].append({
                        "author_id": author_id,
                        "name": author_name[:200],
                        "org": author_org[:300]
                    })
                    
                    result['written_by'].append({
                        "paper_id": paper_id,
                        "author_id": author_id,
                        "author_order": idx
                    })
        
        # Extract keywords
        keywords = paper.get("keywords", [])
        
        # If no keywords, extract from title
        if not keywords and paper.get("title"):
            keywords = self.extract_keywords_from_title(paper.get("title"))
        
        if isinstance(keywords, list):
            for keyword in keywords:
                if not keyword:
                    continue
                keyword = str(keyword)[:100].strip()
                if not keyword:
                    continue
                keyword_id = f"kw_{hash(keyword.lower()) % 10000000}"
                
                result['keywords'].append({
                    "keyword_id": keyword_id,
                    "keyword": keyword
                })
                
                result['has_keyword'].append({
                    "paper_id": paper_id,
                    "keyword_id": keyword_id
                })
        
        # Extract citation relationships
        references = paper.get("references", [])
        if isinstance(references, list):
            for ref_id in references:
                if ref_id:
                    result['cites'].append({
                        "citing_paper": paper_id,
                        "cited_paper": str(ref_id)
                    })
        
        return result
    
    def _write_batch_to_temp(self, results, temp_files, seen_authors, seen_keywords):
        """Write a batch of results to temporary files"""
        # Helper function to safely format CSV row
        def safe_writerow(writer, row):
            """Write row safely, handling all special characters"""
            try:
                writer.writerow(row)
            except Exception as e:
                # If error, try with escapechar
                for key in row:
                    if isinstance(row[key], str):
                        # Replace problematic characters
                        row[key] = str(row[key]).replace('"', '""').replace('\n', ' ').replace('\r', ' ')
                writer.writerow(row)
        
        writers = {
            'papers': csv.DictWriter(temp_files['papers'], 
                fieldnames=['paper_id', 'title', 'abstract', 'year', 'venue', 'n_citation'], 
                quoting=csv.QUOTE_MINIMAL, escapechar='\\'),
            'authors': csv.DictWriter(temp_files['authors'], 
                fieldnames=['author_id', 'name', 'org'],
                quoting=csv.QUOTE_MINIMAL, escapechar='\\'),
            'keywords': csv.DictWriter(temp_files['keywords'], 
                fieldnames=['keyword_id', 'keyword'],
                quoting=csv.QUOTE_MINIMAL, escapechar='\\'),
            'written_by': csv.DictWriter(temp_files['written_by'], 
                fieldnames=['paper_id', 'author_id', 'author_order'],
                quoting=csv.QUOTE_MINIMAL, escapechar='\\'),
            'cites': csv.DictWriter(temp_files['cites'], 
                fieldnames=['citing_paper', 'cited_paper'],
                quoting=csv.QUOTE_MINIMAL, escapechar='\\'),
            'has_keyword': csv.DictWriter(temp_files['has_keyword'], 
                fieldnames=['paper_id', 'keyword_id'],
                quoting=csv.QUOTE_MINIMAL, escapechar='\\')
        }
        
        for result in results:
            if result is None:
                continue
            
            # Write paper
            if result['paper']:
                safe_writerow(writers['papers'], result['paper'])
            
            # Write authors (deduplicate)
            for author in result['authors']:
                if author['author_id'] not in seen_authors:
                    safe_writerow(writers['authors'], author)
                    seen_authors.add(author['author_id'])
            
            # Write keywords (deduplicate)
            for keyword in result['keywords']:
                if keyword['keyword_id'] not in seen_keywords:
                    safe_writerow(writers['keywords'], keyword)
                    seen_keywords.add(keyword['keyword_id'])
            
            # Write relationships
            for rel in result['written_by']:
                safe_writerow(writers['written_by'], rel)
            for rel in result['cites']:
                safe_writerow(writers['cites'], rel)
            for rel in result['has_keyword']:
                safe_writerow(writers['has_keyword'], rel)
    
    def run_pipeline(self, input_file, num_workers=2, batch_size=3000):
        """
        Run complete data pipeline: clean + extract to CSV
        
        Args:
            input_file: Input JSONL file path (raw ACM data)
            num_workers: Number of parallel workers
            batch_size: Number of lines per batch
        """
        print("=" * 70)
        print("📊 Academic Data Processing Pipeline")
        print("=" * 70)
        print(f"Dataset type: {self.dataset_type.upper()}")
        print(f"Output directory: {self.processed_dir}/")
        print(f"Step 1: Remove unnecessary fields")
        print(f"Step 2: Extract entities and relationships to CSV")
        print(f"\nUsing {num_workers} processes for parallel processing...")
        print(f"Batch size: {batch_size:,} lines")
        
        # Count total lines
        print("\nCounting file lines...")
        with open(input_file, "r", encoding="utf-8") as f:
            total_lines = sum(1 for _ in f)
        print(f"Total {total_lines:,} lines of data")
        
        # Initialize temporary files for incremental writing
        temp_dir = tempfile.mkdtemp()
        temp_files = {
            'papers': open(f"{temp_dir}/papers.csv", 'w', encoding='utf-8', newline='', buffering=1024*1024),
            'authors': open(f"{temp_dir}/authors.csv", 'w', encoding='utf-8', newline='', buffering=1024*1024),
            'keywords': open(f"{temp_dir}/keywords.csv", 'w', encoding='utf-8', newline='', buffering=1024*1024),
            'written_by': open(f"{temp_dir}/written_by.csv", 'w', encoding='utf-8', newline='', buffering=1024*1024),
            'cites': open(f"{temp_dir}/cites.csv", 'w', encoding='utf-8', newline='', buffering=1024*1024),
            'has_keyword': open(f"{temp_dir}/has_keyword.csv", 'w', encoding='utf-8', newline='', buffering=1024*1024)
        }
        
        # Write CSV headers
        csv.DictWriter(temp_files['papers'], 
            fieldnames=['paper_id', 'title', 'abstract', 'year', 'venue', 'n_citation'], 
            quoting=csv.QUOTE_MINIMAL, escapechar='\\').writeheader()
        csv.DictWriter(temp_files['authors'], 
            fieldnames=['author_id', 'name', 'org'],
            quoting=csv.QUOTE_MINIMAL, escapechar='\\').writeheader()
        csv.DictWriter(temp_files['keywords'], 
            fieldnames=['keyword_id', 'keyword'],
            quoting=csv.QUOTE_MINIMAL, escapechar='\\').writeheader()
        csv.DictWriter(temp_files['written_by'], 
            fieldnames=['paper_id', 'author_id', 'author_order'],
            quoting=csv.QUOTE_MINIMAL, escapechar='\\').writeheader()
        csv.DictWriter(temp_files['cites'], 
            fieldnames=['citing_paper', 'cited_paper'],
            quoting=csv.QUOTE_MINIMAL, escapechar='\\').writeheader()
        csv.DictWriter(temp_files['has_keyword'], 
            fieldnames=['paper_id', 'keyword_id'],
            quoting=csv.QUOTE_MINIMAL, escapechar='\\').writeheader()
        
        # Keep track of seen IDs for deduplication
        seen_authors = set()
        seen_keywords = set()
        
        # Streaming parallel processing with incremental writing
        print("\nProcessing data (cleaning + extracting)...")
        
        with open(input_file, "r", encoding="utf-8") as infile, \
             mp.Pool(processes=num_workers) as pool:
            
            batch = []
            pbar = tqdm(total=total_lines, desc="Processing", unit="lines")
            
            for line in infile:
                batch.append(line)
                
                # Process when batch reaches specified size
                if len(batch) >= batch_size:
                    results = pool.map(self.process_and_extract, batch, chunksize=50)
                    
                    # Write results immediately to temp files
                    self._write_batch_to_temp(results, temp_files, seen_authors, seen_keywords)
                    
                    pbar.update(len(batch))
                    batch = []
            
            # Process remaining data
            if batch:
                results = pool.map(self.process_and_extract, batch, chunksize=50)
                self._write_batch_to_temp(results, temp_files, seen_authors, seen_keywords)
                pbar.update(len(batch))
            
            pbar.close()
        
        # Close all temp files
        for f in temp_files.values():
            f.close()
        
        # Move temp files to final location
        print("\nFinalizing CSV files...")
        for name in ['papers', 'authors', 'keywords', 'written_by', 'cites', 'has_keyword']:
            shutil.move(f"{temp_dir}/{name}.csv", f"{self.processed_dir}/{name}.csv")
        
        # Clean up temp directory
        os.rmdir(temp_dir)
        
        print(f"\n✅ All CSV files saved to {self.processed_dir}/")
        print(f"\n📊 Processing Summary:")
        print(f"   📄 Processed {total_lines:,} lines")
        print(f"   👥 Unique authors: {len(seen_authors):,}")
        print(f"   🏷️  Unique keywords: {len(seen_keywords):,}")
        
        return True

if __name__ == "__main__":
    import sys
    
    # Check command line argument
    if len(sys.argv) > 1:
        dataset_type = sys.argv[1].lower()
        if dataset_type not in ["train", "test", "combined"]:
            print("Error: dataset_type must be 'train', 'test', or 'combined'")
            print("Usage: python process_data.py [train|test|combined]")
            sys.exit(1)
    else:
        dataset_type = "combined"
    
    # Map dataset type to input file
    input_files = {
        "train": "data/raw/train.jsonl",
        "test": "data/raw/test.jsonl",
        "combined": "data/raw/output.jsonl"
    }
    
    input_file = input_files[dataset_type]
    
    print(f"\n🔍 Processing dataset: {dataset_type.upper()}")
    print(f"📂 Input file: {input_file}")
    
    # Create pipeline with specified dataset type
    pipeline = AcademicDataPipeline(dataset_type=dataset_type)
    
    # Run complete pipeline (adjust num_workers as needed)
    # Low values to prevent memory/disk overload
    pipeline.run_pipeline(input_file, num_workers=8, batch_size=100000)
    
    print("\n" + "=" * 70)
    print("✅ Data pipeline completed successfully!")
    print(f"📁 CSV files saved to: {pipeline.processed_dir}/")
    print("\n📋 Generated CSV files:")
    print("   • papers.csv - Paper information (id, title, abstract, year, venue, n_citation)")
    print("   • authors.csv - Author information (author_id, name, org)")
    print("   • keywords.csv - Keywords (keyword_id, keyword)")
    print("   • written_by.csv - Paper-Author relationships")
    print("   • cites.csv - Citation relationships")
    print("   • has_keyword.csv - Paper-Keyword relationships")
    print("=" * 70)
