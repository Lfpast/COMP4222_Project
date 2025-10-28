import pandas as pd
import json
from tqdm import tqdm
import os

class AcademicDataProcessor:
    def __init__(self):
        self.raw_dir = "data/raw"
        self.processed_dir = "data/processed"
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
    
    def load_raw_data(self):
        """从raw目录加载数据"""
        print(f"\n📂 Loading data from {self.raw_dir}...")
        
        all_papers = []
        
        # 遍历raw目录中的所有数据文件
        for filename in os.listdir(self.raw_dir):
            file_path = os.path.join(self.raw_dir, filename)
            
            if not os.path.isfile(file_path):
                continue
            
            try:
                if filename.endswith('.json'):
                    print(f"📖 Reading JSON file: {filename}")
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                        # 处理不同的JSON格式
                        if isinstance(data, dict) and "papers" in data:
                            all_papers.extend(data["papers"])
                        elif isinstance(data, list):
                            all_papers.extend(data)
                        else:
                            print(f"⚠️  Unknown JSON format in {filename}")
                
                elif filename.endswith('.txt'):
                    print(f"📖 Reading TXT file: {filename}")
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            try:
                                paper = json.loads(line.strip())
                                all_papers.append(paper)
                            except:
                                continue
            
            except Exception as e:
                print(f"❌ Error reading {filename}: {e}")
        
        print(f"✅ Loaded {len(all_papers)} papers")
        return all_papers
    
    def clean_and_transform_data(self):
        """数据清洗和转换 - 保存到data/processed目录"""
        print("\n🔄 Cleaning and transforming data...")
        
        # 加载原始数据
        raw_papers = self.load_raw_data()
        
        if not raw_papers:
            print("❌ No data to process!")
            return False
        
        # 创建实体表
        papers_data = []
        authors_data = {}  # 使用字典去重
        institutions_data = {}
        keywords_data = {}
        
        # 创建关系表
        written_by_data = []
        cites_data = []
        has_keyword_data = []
        
        print(f"📊 Processing {len(raw_papers)} papers...")
        
        for paper in tqdm(raw_papers, desc="Processing papers"):
            # 获取paper_id (可能是id, paper_id, _id等)
            paper_id = paper.get("id") or paper.get("paper_id") or paper.get("_id", "")
            if not paper_id:
                continue
            
            paper_id = str(paper_id)
            
            # 处理论文节点
            papers_data.append({
                "paper_id": paper_id,
                "title": paper.get("title", "")[:500],  # 限制长度
                "abstract": paper.get("abstract", "")[:1000],
                "year": paper.get("year", ""),
                "venue": paper.get("venue", ""),
                "n_citation": paper.get("n_citation", 0)
            })
            
            # 处理作者和写作关系
            authors = paper.get("authors", [])
            if isinstance(authors, list):
                for idx, author in enumerate(authors):
                    if isinstance(author, dict):
                        author_id = str(author.get("id") or author.get("_id", f"author_{hash(author.get('name', ''))% 100000}"))
                        author_name = author.get("name", "Unknown")
                        author_org = author.get("org", "")
                    elif isinstance(author, str):
                        author_id = f"author_{hash(author) % 100000}"
                        author_name = author
                        author_org = ""
                    else:
                        continue
                    
                    # 去重添加作者
                    if author_id not in authors_data:
                        authors_data[author_id] = {
                            "author_id": author_id,
                            "name": author_name[:200],
                            "org": author_org[:200]
                        }
                    
                    # 添加写作关系
                    written_by_data.append({
                        "paper_id": paper_id,
                        "author_id": author_id,
                        "author_order": idx
                    })
            
            # 处理关键词
            keywords = paper.get("keywords", [])
            if isinstance(keywords, list):
                for keyword in keywords:
                    if not keyword:
                        continue
                    keyword = str(keyword)[:100]
                    keyword_id = f"kw_{hash(keyword) % 100000}"
                    
                    # 去重添加关键词
                    if keyword_id not in keywords_data:
                        keywords_data[keyword_id] = {
                            "keyword_id": keyword_id,
                            "keyword": keyword
                        }
                    
                    has_keyword_data.append({
                        "paper_id": paper_id,
                        "keyword_id": keyword_id
                    })
            
            # 处理引用关系
            references = paper.get("references", [])
            if isinstance(references, list):
                for ref_id in references:
                    if ref_id:
                        cites_data.append({
                            "citing_paper": paper_id,
                            "cited_paper": str(ref_id)
                        })
        
        # 保存为CSV文件到processed目录
        print(f"\n💾 Saving processed data to {self.processed_dir}...")
        
        pd.DataFrame(papers_data).drop_duplicates(subset=['paper_id']).to_csv(
            f"{self.processed_dir}/papers.csv", index=False)
        pd.DataFrame(list(authors_data.values())).to_csv(
            f"{self.processed_dir}/authors.csv", index=False)
        pd.DataFrame(list(keywords_data.values())).to_csv(
            f"{self.processed_dir}/keywords.csv", index=False)
        pd.DataFrame(written_by_data).to_csv(
            f"{self.processed_dir}/written_by.csv", index=False)
        pd.DataFrame(cites_data).to_csv(
            f"{self.processed_dir}/cites.csv", index=False)
        pd.DataFrame(has_keyword_data).to_csv(
            f"{self.processed_dir}/has_keyword.csv", index=False)
        
        # 打印统计信息
        print("\n📊 Data Processing Summary:")
        print(f"   📄 Papers: {len(papers_data)}")
        print(f"   👥 Authors: {len(authors_data)}")
        print(f"   🏷️  Keywords: {len(keywords_data)}")
        print(f"   ✍️  Written-by relationships: {len(written_by_data)}")
        print(f"   🔗 Citation relationships: {len(cites_data)}")
        print(f"   🏷️  Keyword relationships: {len(has_keyword_data)}")
        
        print(f"\n✅ CSV files saved to {self.processed_dir}/")
        
        return True

def main():
    """主函数"""
    print("=" * 70)
    print("🔄 Academic Data Processing")
    print("=" * 70)
    
    processor = AcademicDataProcessor()
    
    # 检查raw目录是否有数据
    if not os.path.exists(processor.raw_dir) or not os.listdir(processor.raw_dir):
        print(f"\n❌ No data found in {processor.raw_dir}/")
        print("\n💡 Please run download_data.py first to download data.")
        return 1
    
    # 处理数据
    success = processor.clean_and_transform_data()
    
    print("\n" + "=" * 70)
    if success:
        print("✅ Data processing completed successfully!")
        print(f"📁 Processed data saved to: {processor.processed_dir}/")
        print("\n📋 Generated CSV files:")
        print("   • papers.csv - Paper information")
        print("   • authors.csv - Author information")
        print("   • keywords.csv - Keywords")
        print("   • written_by.csv - Paper-Author relationships")
        print("   • cites.csv - Citation relationships")
        print("   • has_keyword.csv - Paper-Keyword relationships")
        print("\n📋 Next steps:")
        print("   1. Verify the CSV files")
        print("   2. Run neo4j_import.py to import data to Neo4j")
        print("   3. Run han_model.py to train the model")
    else:
        print("❌ Data processing failed!")
        print("\n💡 Troubleshooting:")
        print("   1. Check if data files exist in data/raw/")
        print("   2. Verify data file format (JSON or TXT)")
        print("   3. Check for errors in the output above")
    print("=" * 70)
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
