import pandas as pd
import os
import json
import csv
from tqdm import tqdm
import re

class JSONSubGraphFilter:
    """
    Reads a large raw JSON file (one paper per line), identifies a
    focused subgraph (seeds + 1-hop), and exports that exact subgraph
    into the 6 CSV files required by 'neo4j_import.py'.

    This avoids the need to import the entire 2M+ papers into Neo4j.
    """
    
    def __init__(self, raw_json_path, output_dir="data/focused_v1", 
                 venue_list=None, years=None):
        
        self.raw_json_path = raw_json_path
        self.output_dir = output_dir
        
        self.seed_years = years or [2020, 2021, 2022]
        
        # Fuzzy venue list. We use lowercase and 'contains'
        self.seed_venue_list = venue_list or [
            "cvpr", "computer vision and pattern recognition",
            "iclr", "international conference on learning representations",
            "icml", "international conference on machine learning"
        ]
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"📁 Created output directory: {self.output_dir}")
            
        # These will be built in Pass 1
        self.seed_ids = set()
        self.cites_map = {}  # key: paper_id, val: set(ref_ids)
        self.cited_by_map = {} # key: paper_id, val: set(citer_ids)
        
        # This will be built after Pass 1
        self.final_paper_ids = set()

    def _venue_match(self, paper_venue, paper_year):
        """
        Checks if a paper matches our seed criteria.
        """
        if paper_year not in self.seed_years:
            return False
        
        if not paper_venue or not isinstance(paper_venue, str):
            return False
            
        venue_lower = paper_venue.lower()
        for venue_str in self.seed_venue_list:
            if venue_str in venue_lower:
                return True
        return False

    def _get_author_id(self, author):
        """
        Gets a unique author ID. Uses 'id' if available,
        otherwise falls back to 'name'.
        """
        if author.get('id'):
            return str(author['id'])
        # Fallback to name if ID is missing or empty
        return str(author.get('name', ''))

    def _get_keyword_id(self, kw):
        """
        Uses the keyword string itself as the unique ID.
        """
        return str(kw)

    def run_pass_1_indexing(self):
        """
        First pass: Read the entire JSON file to:
        1. Build the full 'cites' and 'cited_by' maps (graph index).
        2. Identify all 'seed' paper IDs.
        """
        print("=" * 70)
        print(f"🚀 Starting Pass 1: Indexing graph from {self.raw_json_path}")
        print("   This may take a while, but it's much faster than Neo4j.")
        
        self.cites_map = {}
        self.cited_by_map = {}
        self.seed_ids = set()

        line_count = 0
        with open(self.raw_json_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Pass 1/2: Indexing"):
                try:
                    paper = json.loads(line)
                    paper_id = str(paper['id'])
                    
                    # 1. Check for seed paper
                    if self._venue_match(paper.get('venue'), paper.get('year')):
                        self.seed_ids.add(paper_id)
                        
                    # 2. Build cites_map (forward links)
                    references = paper.get('references', [])
                    if references:
                        ref_ids = set(str(ref) for ref in references)
                        self.cites_map[paper_id] = ref_ids
                    
                    # 3. Build cited_by_map (backward links)
                    for ref_id in references:
                        ref_id_str = str(ref_id)
                        if ref_id_str not in self.cited_by_map:
                            self.cited_by_map[ref_id_str] = set()
                        self.cited_by_map[ref_id_str].add(paper_id)
                        
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Warning: Skipping malformed line {line_count}: {e}")
                line_count += 1
                
        print(f"   ✅ Pass 1 complete.")
        print(f"   Found {len(self.seed_ids):,} seed papers.")
        print(f"   Indexed {len(self.cites_map):,} papers with references.")
        print(f"   Indexed {len(self.cited_by_map):,} papers with citations.")

    def run_neighborhood_expansion(self):
        """
        Uses the in-memory maps from Pass 1 to find the full
        1-hop neighborhood and create the final set of paper IDs.
        """
        print("=" * 70)
        print("🚀 Starting Neighborhood Expansion (in-memory)")
        
        self.final_paper_ids = set(self.seed_ids)
        
        with tqdm(self.seed_ids, desc="Expanding neighborhood") as pbar:
            for seed_id in pbar:
                # 1. Get papers CITED BY seed (forward)
                if seed_id in self.cites_map:
                    self.final_paper_ids.update(self.cites_map[seed_id])
                
                # 2. Get papers that CITE seed (backward)
                if seed_id in self.cited_by_map:
                    self.final_paper_ids.update(self.cited_by_map[seed_id])
        
        print(f"   ✅ Expansion complete.")
        print(f"   Total papers in focused subgraph (seeds + 1-hop): {len(self.final_paper_ids):,}")

    def run_pass_2_exporting(self):
        """
        Second pass: Read the JSON again and filter, writing
        only the data for papers in 'final_paper_ids' to the 6 CSVs.
        """
        print("=" * 70)
        print(f"🚀 Starting Pass 2: Exporting focused subgraph to {self.output_dir}")
        
        # Keep track of which nodes we've already saved to avoid duplicates
        seen_authors = set()
        seen_keywords = set()
        
        # Define CSV headers
        headers = {
            "papers": ["paper_id", "title", "abstract", "year", "venue", "n_citation"],
            "authors": ["author_id", "name", "org"],
            "keywords": ["keyword_id", "keyword"],
            "cites": ["citing_paper", "cited_paper"],
            "written_by": ["paper_id", "author_id", "author_order"],
            "has_keyword": ["paper_id", "keyword_id"]
        }
        
        # Open 6 file writers
        f_papers = open(f"{self.output_dir}/papers.csv", 'w', encoding='utf-8', newline='')
        w_papers = csv.DictWriter(f_papers, headers["papers"])
        w_papers.writeheader()

        f_authors = open(f"{self.output_dir}/authors.csv", 'w', encoding='utf-8', newline='')
        w_authors = csv.DictWriter(f_authors, headers["authors"])
        w_authors.writeheader()

        f_keywords = open(f"{self.output_dir}/keywords.csv", 'w', encoding='utf-8', newline='')
        w_keywords = csv.DictWriter(f_keywords, headers["keywords"])
        w_keywords.writeheader()

        f_cites = open(f"{self.output_dir}/cites.csv", 'w', encoding='utf-8', newline='')
        w_cites = csv.DictWriter(f_cites, headers["cites"])
        w_cites.writeheader()

        f_written_by = open(f"{self.output_dir}/written_by.csv", 'w', encoding='utf-8', newline='')
        w_written_by = csv.DictWriter(f_written_by, headers["written_by"])
        w_written_by.writeheader()

        f_has_keyword = open(f"{self.output_dir}/has_keyword.csv", 'w', encoding='utf-8', newline='')
        w_has_keyword = csv.DictWriter(f_has_keyword, headers["has_keyword"])
        w_has_keyword.writeheader()

        line_count = 0
        papers_written = 0
        try:
            with open(self.raw_json_path, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc="Pass 2/2: Exporting"):
                    try:
                        paper = json.loads(line)
                        paper_id = str(paper['id'])
                        
                        # The CORE filtering logic:
                        if paper_id not in self.final_paper_ids:
                            continue
                        
                        papers_written += 1
                        
                        # 1. Write to papers.csv
                        paper_data = {
                            "paper_id": paper_id,
                            "title": paper.get('title', ''),
                            "abstract": paper.get('abstract', ''),
                            "year": paper.get('year', 0),
                            "venue": paper.get('venue', ''),
                            "n_citation": paper.get('n_citation', 0)
                        }
                        w_papers.writerow(paper_data)
                        
                        # 2. Write to authors.csv and written_by.csv
                        for i, author in enumerate(paper.get('authors', [])):
                            author_id = self._get_author_id(author)
                            if not author_id: # Skip if no ID and no name
                                continue

                            # Write author node (if new)
                            if author_id not in seen_authors:
                                w_authors.writerow({
                                    "author_id": author_id,
                                    "name": author.get('name', ''),
                                    "org": author.get('org', '')
                                })
                                seen_authors.add(author_id)
                            
                            # Write paper-author edge
                            w_written_by.writerow({
                                "paper_id": paper_id,
                                "author_id": author_id,
                                "author_order": i
                            })
                            
                        # 3. Write to keywords.csv and has_keyword.csv
                        for kw in paper.get('keywords', []):
                            keyword_id = self._get_keyword_id(kw)
                            if not keyword_id:
                                continue
                                
                            # Write keyword node (if new)
                            if keyword_id not in seen_keywords:
                                w_keywords.writerow({
                                    "keyword_id": keyword_id,
                                    "keyword": keyword_id # Use keyword itself as ID
                                })
                                seen_keywords.add(keyword_id)
                            
                            # Write paper-keyword edge
                            w_has_keyword.writerow({
                                "paper_id": paper_id,
                                "keyword_id": keyword_id
                            })
                            
                        # 4. Write to cites.csv
                        for ref_id in paper.get('references', []):
                            ref_id_str = str(ref_id)
                            # CRITICAL: Only write link if the cited paper
                            # is ALSO in our final subgraph
                            if ref_id_str in self.final_paper_ids:
                                w_cites.writerow({
                                    "citing_paper": paper_id,
                                    "cited_paper": ref_id_str
                                })
                                
                    except (json.JSONDecodeError, KeyError) as e:
                        print(f"Warning: Skipping malformed line {line_count}: {e}")
                    line_count += 1
        
        finally:
            # Close all file handles
            f_papers.close()
            f_authors.close()
            f_keywords.close()
            f_cites.close()
            f_written_by.close()
            f_has_keyword.close()

        print(f"   ✅ Pass 2 complete.")
        print(f"   Wrote {papers_written:,} papers to {self.output_dir}/papers.csv")
        print(f"   Wrote {len(seen_authors):,} unique authors to {self.output_dir}/authors.csv")
        print(f"   Wrote {len(seen_keywords):,} unique keywords to {self.output_dir}/keywords.csv")

    def run_filter(self):
        """
        Executes the full two-pass filtering workflow.
        """
        self.run_pass_1_indexing()
        self.run_neighborhood_expansion()
        self.run_pass_2_exporting()


if __name__ == "__main__":
    
    # --- Configuration ---
    
    # 1. Point this to your GIANT raw JSON file
    #    (e.g., dblp_v10.json or whatever it's called)
    RAW_JSON_PATH = "D:/path/to/your/dblp_raw_data.json" # <-- IMPORTANT: Update this
    
    # 2. This is where the new, small CSVs will be saved
    OUTPUT_DIR = "data/focused_v1"
    
    # 3. Years for the seed papers
    SEED_YEARS = [2020, 2021, 2022]
    
    # 4. Fuzzy venue list (lowercase)
    SEED_VENUE_LIST = [
        "cvpr", "computer vision and pattern recognition",
        "iclr", "international conference on learning representations",
        "icml", "international conference on machine learning"
    ]
    
    # ---------------------
    
    if not os.path.exists(RAW_JSON_PATH):
        print(f"❌ Error: Raw JSON file not found at: {RAW_JSON_PATH}")
        print("   Please update the 'RAW_JSON_PATH' variable in this script.")
    else:
        print("=" * 70)
        print("🚀 JSON Subgraph Filter to CSV")
        print("=" * 70)
        print(f"   Reading from: {RAW_JSON_PATH}")
        print(f"   Writing to:   {OUTPUT_DIR}")
        print(f"   Seed Venues:  {', '.join(SEED_VENUE_LIST)}")
        print(f"   Seed Years:   {SEED_YEARS}")
        
        try:
            filterer = JSONSubGraphFilter(
                raw_json_path=RAW_JSON_PATH,
                output_dir=OUTPUT_DIR,
                venue_list=SEED_VENUE_LIST,
                years=SEED_YEARS
            )
            
            filterer.run_filter()
            
            print("\n" + "=" * 70)
            print("🎉 NEXT STEPS")
            print("=" * 70)
            print("   1. Your new, focused CSVs are now in the 'data/focused_v1/' directory.")
            print("   2. Open your 'neo4j_import.py' script.")
            print("   3. In its '__main__' block, change 'dataset_type' to 'focused_v1'")
            print("      (This will make it read from 'data/focused_v1/').")
            print("   4. Run 'neo4j_import.py' to import this new, fast, focused dataset.")
            print("   5. Run 'han_model.py' to train your embeddings.")
            
        except Exception as e:
            print(f"\n❌ Filtering failed: {e}")
            import traceback
            traceback.print_exc()