import os
import requests
import zipfile
from tqdm import tqdm

class AminerDataDownloader:
    def __init__(self):
        self.base_dir = "data/raw"
        os.makedirs(self.base_dir, exist_ok=True)
        
        # Aminer数据集URLs
        self.dataset_urls = {
            "citation_network": "https://aminer.org/lab-datasets/citation/citation-network1.zip",
            "dblp_citation": "https://aminer.org/lab-datasets/citation/DBLP_Citation_2014.zip", 
            "aminer_sample": "https://aminer.org/lab-datasets/aminer/aminer_sample.zip"
        }
    
    def download_file(self, url, filename):
        """下载文件并显示进度条"""
        print(f"Downloading {filename} from {url}")
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()  # 如果请求失败则抛出异常
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filename, 'wb') as file, tqdm(
                desc=filename,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(chunk_size=1024):
                    size = file.write(data)
                    bar.update(size)
            
            return True
        except Exception as e:
            print(f"Download failed: {e}")
            return False
    
    def extract_zip(self, zip_path, extract_to):
        """解压ZIP文件"""
        print(f"Extracting {zip_path}")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            return True
        except Exception as e:
            print(f"❌ Extraction failed: {e}")
            return False
    
    def download_aminer_citation_data(self):
        """下载Aminer引用网络数据"""
        url = self.dataset_urls["citation_network"]
        zip_path = os.path.join(self.base_dir, "citation_network.zip")
        
        # 下载文件
        download_success = self.download_file(url, zip_path)
        if not download_success:
            return False
        
        # 解压文件
        extract_success = self.extract_zip(zip_path, self.base_dir)
        if not extract_success:
            return False
        
        print("Aminer citation data downloaded and extracted successfully!")
        return True
    
    def check_download_success(self):
        """检查下载是否成功"""
        print("Checking download success...")
        
        # 检查ZIP文件是否存在
        zip_path = os.path.join(self.base_dir, "citation_network.zip")
        if not os.path.exists(zip_path):
            print("ZIP file not found")
            return False
        
        # 检查解压后的文件
        extracted_files = os.listdir(self.base_dir)
        data_files = [f for f in extracted_files if f.endswith('.json') or f.endswith('.txt')]
        
        if not data_files:
            print("No data files found after extraction")
            return False
        
        print(f"Found {len(data_files)} data files:")
        for file in data_files:
            file_path = os.path.join(self.base_dir, file)
            file_size = os.path.getsize(file_path)
            print(f"   - {file} ({file_size} bytes)")
        
        return True

def main():
    """主函数"""
    downloader = AminerDataDownloader()
    
    print("Starting Aminer Dataset Download...")
    
    # 下载Aminer数据
    success = downloader.download_aminer_citation_data()
    
    # 检查下载是否成功
    if success:
        success = downloader.check_download_success()
    
    if success:
        print("\nDownload completed successfully!")
        print(f"Data saved to: {downloader.base_dir}")
    else:
        print("\nDownload failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())