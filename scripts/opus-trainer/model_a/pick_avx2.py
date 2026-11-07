#!/usr/bin/env python3
"""Like vast_pick.py but prints the host CPU and drops CPUs without AVX2. The marian-cuda image's marian binary dies with
'Illegal instruction' on an Intel Xeon E7-4880 v2 (Ivy Bridge, AVX only) - found 2026-11-06 by the first gpu_smoke on box 59381897."""
import argparse, re, sys
sys.path.insert(0, "/tmp/km-smoke/gpu-session/helpers")
import vast_pick as vp
ns = argparse.Namespace(limit=80, allow_blackwell=False, gpu_ram_min=20000, gpu_ram_max=26000, disk_gb=80, min_inet_down=200, min_reliability=0.98,
                        min_cuda=11.8, base_url="https://console.vast.ai/api/v0", hours=7, inbound_gb=3)
offers = vp.search_offers(ns)
OLD = re.compile(r"(E[357]-\d+\s*(v[12])?\b(?!.*v[34])|E5-\d+\s*v2|E5-\d+\s*0?\b|X5[0-9]{3}|E7-|Xeon\(R\)\s+CPU\s+E5-\d{4}\s*v[12]?\b|Gold 51|i[357]-[1-3]\d{3}\b)", re.I)
good = []
for o in offers:
    cpu = o.get("cpu_name") or "?"
    old = bool(re.search(r"E7-|E5-\d{4}\s*(v2)?\s*@|E5-\d{4}\s*v2|X5\d{3}|X3\d{3}|E3-\d{4}\s*(v2)?\s*@", cpu)) and not re.search(r"v[34]", cpu)
    if "ssh" in sys.argv and False: pass
    c = vp.session_cost(o, disk_gb=80, hours=7, inbound_gb=3)
    good.append((c["hourly_total"], o, cpu, old))
good.sort(key=lambda x: x[0])
for h, o, cpu, old in good[:25]:
    print(f"{'OLD?' if old else 'ok  '} id={o['id']} {o.get('gpu_name')} {o.get('geolocation')} ${h:.4f}/h rel={o.get('reliability2'):.4f} down={o.get('inet_down'):.0f} cpu={cpu} cores={o.get('cpu_cores_effective')} ram={o.get('cpu_ram')}")
