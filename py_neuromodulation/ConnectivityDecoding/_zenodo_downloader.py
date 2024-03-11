import wget

record_id = "10804702"
file_name = "AGPotData.csv"

wget.download(
    f"https://zenodo.org/api/records/{record_id}/files/{file_name}/content",
    out=f"{file_name}.mat",
)
