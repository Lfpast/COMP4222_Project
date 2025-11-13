#!/usr/bin/env python3
"""Test Neo4j connection with different URIs"""

from py2neo import Graph
import sys

# Different URI formats to try
uris_to_test = [
    "bolt://localhost:7687",
    "bolt://127.0.0.1:7687",
    "neo4j://localhost:7687",
    "neo4j://127.0.0.1:7687",
    "bolt://$(hostname).local:7687",  # WSL to Windows host
    "bolt://host.docker.internal:7687",  # Docker/WSL host
]

# You can also try Windows host IP
import socket
try:
    # Get WSL's default gateway (usually Windows host IP)
    import subprocess
    result = subprocess.run(['ip', 'route', 'show', 'default'], 
                          capture_output=True, text=True)
    if result.returncode == 0:
        windows_ip = result.stdout.split()[2]
        uris_to_test.append(f"bolt://{windows_ip}:7687")
        print(f"💡 Detected Windows host IP: {windows_ip}")
except:
    pass

username = "neo4j"
password = "12345678"  # Change to your password

print("=" * 70)
print("🔍 Testing Neo4j Connections")
print("=" * 70)
print(f"Username: {username}")
print(f"Password: {'*' * len(password)}")
print()

successful_uri = None

for uri in uris_to_test:
    print(f"Testing: {uri}...", end=" ")
    try:
        graph = Graph(uri, auth=(username, password))
        result = graph.run("RETURN 1 as test").data()
        if result and result[0]['test'] == 1:
            print("✅ SUCCESS!")
            successful_uri = uri
            break
    except Exception as e:
        print(f"❌ Failed: {str(e)[:50]}...")

print()
if successful_uri:
    print("=" * 70)
    print(f"✅ Working URI found: {successful_uri}")
    print("=" * 70)
    print("\nUpdate neo4j_import.py with this URI:")
    print(f'NEO4J_URI = "{successful_uri}"')
else:
    print("=" * 70)
    print("❌ No working connection found")
    print("=" * 70)
    print("\n💡 Troubleshooting:")
    print("1. If Neo4j is running on Windows (not WSL):")
    print("   - Get Windows IP: Run 'ipconfig' in Windows CMD")
    print("   - Look for 'Ethernet adapter vEthernet (WSL)'")
    print("   - Use that IP in URI: bolt://<WINDOWS_IP>:7687")
    print()
    print("2. Check Neo4j is actually running:")
    print("   - Open Neo4j Desktop")
    print("   - Check database is started")
    print("   - Check connection details in Neo4j Desktop")
    print()
    print("3. Check firewall:")
    print("   - Windows Firewall might be blocking WSL connections")
    print("   - Try disabling temporarily for testing")
    print()
    print("4. Alternative: Run Neo4j in WSL directly")
    
    sys.exit(1)
