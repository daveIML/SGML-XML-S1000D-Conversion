import shutil
for exe in ("openai", "onsgmls", "sx"):
    print(f"{exe}: {shutil.which(exe)}")