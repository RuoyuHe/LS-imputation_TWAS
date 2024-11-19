#### here we get some snps for the type I case
#### we randomly extract ~3000 SNPs after pruning, these snps are used to generate Y regardless of the association between gene exp and Y
#### call it effect snps

library(data.table)

bed_path = '/scratch.global/he000176/lipA/bed/pruned/'
total_snp = NULL
total_bim = NULL
save_path = '/home/panwei/he000176/deepRIV/ImputedTraits/data/for_revision/'

size = 136

tmp_path = '/scratch.global/he000176/UKB/tmp/'
sample_ids = fread('/home/panwei/he000176/deepRIV/ImputedTraits/data/1003_ID.txt')$V1

for(chr in 1:22){
    print(chr)
    current_bim = fread(paste0(bed_path,'/chr',chr,'.bim'))
    colnames(current_bim) = c('chr','id','tmp','pos','A1','A2')
    # remove all rows with no rs ids
    current_bim = current_bim[grep("^rs", id), ]
    
    idx = sample(1:nrow(current_bim), size)
        
    tmp_rs_path = paste0(tmp_path,'/chr',chr,'_rs.txt')
    fwrite(data.frame(ID = current_bim$id[idx], REF = current_bim$A1[idx]), tmp_rs_path, sep=' ', col.names=F)
    
    ###UKB
    prefix = paste0('/home/panwei/shared/UKBiobankIndiv/imputed/pgen/ukbb_chr',chr,'_1')
    ukb_plink_command = paste0('plink2 --pfile ',prefix,
                             ' --chr ', chr,
                             ' --extract ',tmp_rs_path,
                             ' --export A include-alt',
                             ' --export-allele ',tmp_rs_path,
                             ' --snps-only just-acgt',
                             ' --out ',paste0(tmp_path,'/chr',chr))
    ukb_plink_msg = system(ukb_plink_command,intern=TRUE)

    ukb_snp = list()
    ukb_snp$bed = tryCatch(fread(paste0(tmp_path,'/chr',chr,'.raw'),data.table=F),
                     error=function(e){cat("ERROR :",
                                           conditionMessage(e),
                                           "\n")})

    ukb_snp$fam = ukb_snp$bed[,c(1:6)] 
    # create bim file without pos
    tmp_ukb_names = colnames(ukb_snp$bed)[-c(1:6)]
    ukb_snp$bim = data.frame(ID = sub("_.*", "", tmp_ukb_names), 
                           REF = sub(".*_(.*)\\(/.*", "\\1", tmp_ukb_names), 
                           ALT = sub(".*\\(/(.*)\\)", "\\1", tmp_ukb_names))
    # check for special characters in rs id, REF and ALT alleles
    unique(unlist(strsplit(paste(ukb_snp$bim$ID,collapse=''),split='')))
    unique(unlist(strsplit(paste(ukb_snp$bim$REF,collapse=''),split='')))
    unique(unlist(strsplit(paste(ukb_snp$bim$ALT,collapse=''),split='')))

    # turn ukb_snp$bed into an R matrix
    ukb_snp$bed = as.matrix(ukb_snp$bed[,-c(1:6)])
    colnames(ukb_snp$bed) = ukb_snp$bim$ID
    rownames(ukb_snp$bed) = ukb_snp$fam$IID

    rm(tmp_ukb_names)
    
    idx = match(sample_ids, ukb_snp$fam$IID)
    idx = na.omit(idx)
    FAM = ukb_snp$fam[idx,]
    cat('all ids match? ',all(sample_ids==FAM$IID),'\n')
    sample_ids = FAM$IID
    cat('length of sample id idx: ',length(idx),'\n')
    
    
    if(is.null(total_snp)){
        total_snp = ukb_snp$bed[idx,]
    }else{
        total_snp = cbind(total_snp, ukb_snp$bed[idx,])
    }

    if(is.null(total_bim)){
        total_bim = ukb_snp$bim
    }else{
        total_bim = rbind(total_bim, ukb_snp$bim)
    }
    
    if(chr==1){
        fwrite(data.frame(ID=FAM$IID), paste0(save_path,'/effect_snps_ID.txt'), col.names=F)
    }
}

gene_inds = fread('/home/panwei/he000176/deepRIV/ImputedTraits/data/gene_ind.txt',header=F)
a = fread(paste0('/home/panwei/he000176/deepRIV/ImputedTraits/data/',gene_inds$V1[1],'_rs.txt'),header=F)

for(i in 2:nrow(gene_inds)){
    a = rbind(a, fread(paste0('/home/panwei/he000176/deepRIV/ImputedTraits/data/',gene_inds$V1[i],'_rs.txt'),header=F))
}

if (sum(total_bim$ID %in% a$V1) > 0) {
    tmp_idx <- which(total_bim$ID %in% a$V1)
    if (length(tmp_idx) > 0) {
        total_bim <- total_bim[-tmp_idx, ,drop=FALSE]   
        total_snp <- total_snp[, -tmp_idx,drop=FALSE] 
    }
}

fwrite(total_snp, paste0(save_path,'/effect_snps.txt'), sep=' ')
fwrite(total_bim, paste0(save_path,'/effect_snps.bim'), sep=' ')

