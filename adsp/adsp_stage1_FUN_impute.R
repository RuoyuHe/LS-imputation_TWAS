library(BEDMatrix)
library(dplyr)
library(tidyr)
library(igraph)
library(stringi)
library(data.table)

#setwd("~/deepRIV/UKB/data")
source("~/deepRIV/UKB/code/allele_qc.R")

# useful helper functions
get_gene_name <- function(results){
  gene_gtf = rtracklayer::import('/home/panwei/shared/GTEx_v8/gencode.v26.GRCh38.genes.gtf')
  gene_df <- data.frame(as.data.frame(gene_gtf)[c("gene_id", "gene_name")])
  gene_df <- gene_df[!duplicated(gene_df), ]
  results = merge(results, gene_df, by.x='gene_names', by.y='gene_id',
                      all.x=T, all.y=F, sort=F)
  return(results)
}

match_prot_gene <-function(results){
    annot = fread('/home/panwei/he000176/deepRIV/ImputedTraits/code/adsp/olink_annotations.tsv')
    prot_annot = annot[,c('uniprot_id','protein_name','gene_symbol')]
    colnames(prot_annot) = c('gene_names','protein_name','gene_name')
    results = merge(results, prot_annot, by = 'gene_names')
    return(results)
}

tmp_dirs_list = list()

generate_random_dir_name <- function(length = 16) {
  random_string <- stri_rand_strings(1, length, pattern = "[a-zA-Z0-9]")
  return(paste0("dir_", random_string))
}

create_random_directory <- function(path = '/scratch.global/he000176/') {
  if (!dir.exists(path)){
      dir.create(path)    
  }
    
  dir_name <- generate_random_dir_name()
  dir_name <- paste0(path, dir_name,'/')
  
  # Check if directory exists, regenerate name if needed
  while (dir.exists(dir_name)) {
    dir_name <- generate_random_dir_name()
    dir_name <- paste0(path, dir_name,'/')
  }
  
  # Add to the global list with names
  tmp_dirs_list[[dir_name]] <- dir_name
  assign("tmp_dirs_list", tmp_dirs_list, envir = .GlobalEnv)
    
  # Create the directory
  dir.create(dir_name)
  return(dir_name)
}

mean_imputation <- function(data) {
  impute_column <- function(col) {
    na_indices <- which(is.na(col))
    if (length(na_indices) > 0) {
      mean_value <- mean(col, na.rm = TRUE)
      col[na_indices] <- mean_value
    }
    return(col)
  }
  
  imputed_data <- apply(data, 2, impute_column)
  return(data.frame(imputed_data))
}


##  # Covariates
#COV5TABLE = fread("~/deepRIV/UKB/data/GTEx/Whole_Blood.v8.covariates.txt",header = T,data.table=F)
COV5TABLE = fread("/home/panwei/shared/GTEx_v8/GTEx_Analysis_v8_expression_EUR/expression_covariates/Whole_Blood.v8.EUR.covariates.txt",header = T,
                  data.table=F)
# convert the cov table so its (n_samples, n_covs)
ID = COV5TABLE$ID
transposed_data = t(COV5TABLE[, -1])
# Convert the transposed data to a data frame
new_df = data.frame(transposed_data)
colnames(new_df) = ID
new_df = cbind(ID = rownames(new_df), new_df)
rownames(new_df) = NULL
COV5TABLE = new_df

### Gene Expression
# GENE_EXP = fread("~/deepRIV/UKB/data/GTEx/Whole_Blood.v8.normalized_expression.bed.gz")
GENE_EXP = fread("/home/panwei/shared/GTEx_v8/GTEx_Analysis_v8_expression_EUR/expression_matrices/Whole_Blood.v8.EUR.normalized_expression.bed.gz")
colnames(GENE_EXP)[1] = 'chr'
# whole_blood_buddy = '/home/panwei/he000176/deepRIV/UKB/data/Whole_Blood_buddy.txt'
whole_blood_buddy = '/home/panwei/he000176/deepRIV/UKB/data/Whole_Blood_buddy_EUR.txt'


### ukb phenotype
# white_unrelated_keep = fread('~/deepRIV/UKB/data/white_unrelated_keep_ind.txt',
#                              header = F)
# adsp_pheno_all = fread('~/deepRIV/ImputedTraits/ADSP/ADSPIntegratedPhenotypes_DS_2023.08.08.csv',
#                        data.table = F)

### NEW ###
adsp_pheno_all = fread('~/deepRIV/ImputedTraits/ADSP/adsp_cov.csv',
                       data.table = F)

# keep_idx = sort(na.omit(match(white_unrelated_keep$V1, ukb_pheno_all$f.eid)))
# ukb_pheno_all = ukb_pheno_all[keep_idx,]
# rm(keep_idx)


stage1 <- function(gene_ind,
                   pheno_study = 'igap',
                   save_models = FALSE, 
                   z_path = NULL, 
                   beta_path = NULL,
                   theta_path = NULL,
                   y_path = NULL){
  # get covariates
  cov5table = COV5TABLE
  # get gene expression
  gene_exp = GENE_EXP[gene_ind,]
  chr = gene_exp$chr[1]
  chr = as.numeric(gsub("[^0-9.-]", "", chr))
  
  ###UKB
  prefix = paste0('/home/panwei/shared/UKBiobankIndiv/imputed/pgen/ukbb_chr',chr,'_1')
  
  # start analysis ----------------------------------------------------------
  if(pheno_study == 'igap'){
    # igap imputed
    ukb_pheno_all = fread('~/deepRIV/ImputedTraits/ADSP/igap_imputed_ad.txt',data.table = F)
    ukb_pheno = data.frame(f.eid = as.integer(ukb_pheno_all$FID), pheno = ukb_pheno_all$a40)
  }else if(pheno_study == 'eadb'){
    # eadb imputed
    ukb_pheno_all = fread('~/deepRIV/ImputedTraits/ADSP/eadb_imputed_ad.txt',data.table = F)
    ukb_pheno = data.frame(f.eid = as.integer(ukb_pheno_all$FID), pheno = ukb_pheno_all$e40)
  }else if(pheno_study == 'proxyAD'){
    ukb_pheno_all = fread('~/deepRIV/ImputedTraits/ADSP/igap_imputed_ad.txt',data.table = F)
    ukb_pheno = data.frame(f.eid = as.integer(ukb_pheno_all$FID), pheno = ukb_pheno_all$testtrue - 1)
  }
  adsp_pheno = data.frame(id = adsp_pheno_all$SampleID, pheno = adsp_pheno_all$DX_harmonized)
  adsp_pheno = na.omit(adsp_pheno)
  
  gene_name = as.character(gene_exp$gene_id[1])
  start = floor(gene_exp$start[1]/1000) - 100
  end = ceiling(gene_exp$end[1]/1000) + 100
  
  # generate random directories for GTEx and UKB
  GTEx_dir = create_random_directory()
  UKB_dir = create_random_directory()
  adsp_dir = create_random_directory()
  print(GTEx_dir)
  print(UKB_dir)
  print(adsp_dir)
        
  plink_command = paste0("module load plink/2.00-alpha-091019; \n","plink2 --bfile /home/panwei/he000176/deepRIV/UKB/data/GTEx/GTEx_Analysis_2017-06-05_v8_WholeGenomeSeq_838Indiv_Analysis_Freeze.SHAPEIT2_phased_MAF01",
                         " --chr ",chr," --from-kb ",start," --to-kb ",end,
                         " --keep ",whole_blood_buddy,
                         " --geno 0 --maf 0.05 --hwe 0.001  ",
                         " --export A include-alt",
                         " --snps-only just-acgt",
                         " --out ",GTEx_dir,gene_name
                        )
  plink_msg=system(plink_command,intern=TRUE)  

  snp = list()
  snp$bed = tryCatch(fread(paste0(GTEx_dir,gene_name,'.raw'),data.table=F),
                     error=function(e){cat("ERROR :",
                                       conditionMessage(e),
                                       "\n")})
  if(length(snp)==0)
  {
    FINAL_RESULT = list('NoValue')
    unlink(GTEx_dir, recursive = TRUE)
    unlink(UKB_dir, recursive = TRUE)
    unlink(adsp_dir, recursive = TRUE)
    return(FINAL_RESULT)
  }
  
  snp$fam = snp$bed[,c(1:6)] 
  # create bim file without pos
  tmp_gtex_names = colnames(snp$bed)[-c(1:6)]
  snp$bim = data.frame(ID = sub("_[ACGT]+\\(.*", "", tmp_gtex_names), 
                       REF = sub(".*_b38_([ACGT]+)\\(.*", "\\1", tmp_gtex_names), 
                       ALT = sub(".*\\(/(.*)\\)", "\\1", tmp_gtex_names)
                      )
  # check for special characters in rs id, REF and ALT alleles
  unique(unlist(strsplit(paste(snp$bim$ID,collapse=''),split='')))
  unique(unlist(strsplit(paste(snp$bim$REF,collapse=''),split='')))
  unique(unlist(strsplit(paste(snp$bim$ALT,collapse=''),split='')))

  # turn snp$bed into an R matrix
  snp$bed = as.matrix(snp$bed[,-c(1:6)])
  colnames(snp$bed) = snp$bim$ID
  rownames(snp$bed) = snp$fam$IID

  unlink(GTEx_dir, recursive = TRUE)  
  
  write.table(snp$bim$ID,paste0(UKB_dir,gene_name,'_rs.txt'),row.names=F,quote=F,col.names=F)
  
  grep_command = paste0("zgrep -F -m ",length(snp$bim$ID)," -f ",UKB_dir,gene_name,"_rs.txt /home/panwei/he000176/deepRIV/UKB/data/GTEx/GTEx_Analysis_2017-06-05_v8_WholeGenomeSeq_838Indiv_Analysis_Freeze.lookup_table.txt  > ",UKB_dir,gene_name,"_rs1.txt")
  system(grep_command)
  gtex_dic = read.table(paste0(UKB_dir,gene_name,'_rs1.txt'))
  write.table(data.frame(ID=gtex_dic$V7,REF=snp$bim$REF),paste0(UKB_dir,gene_name,'_rs.txt'),row.names=F,quote=F,col.names=F)
  
  
  ukb_out = paste0(UKB_dir,gene_name)
  ukb_plink_command = paste0('module load plink/2.00-alpha-091019; plink2 --pfile ',prefix,
                             ' --chr ', chr,
                             ' --extract ',UKB_dir,gene_name,'_rs.txt',
                             ' --maf 0.05 --hwe 0.001  ',
                             ' --export A include-alt',
                             ' --export-allele ', UKB_dir,gene_name,'_rs.txt',
                             ' --snps-only just-acgt',
                             ' --out ',ukb_out)
  ukb_plink_msg = system(ukb_plink_command,intern=TRUE)
  
  ukb_snp = list()
  ukb_snp$bed = tryCatch(fread(paste0(ukb_out,'.raw'),data.table=F),
                     error=function(e){cat("ERROR :",
                                           conditionMessage(e),
                                           "\n")})
  if(length(ukb_snp)==0)
  {
    FINAL_RESULT = list('NoValue')
    return(FINAL_RESULT)
  }

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

  # remove tmp dir
  unlink(UKB_dir, recursive = TRUE)
  rm(tmp_ukb_names)
    
  
  # adsp snps
  ########## UPDATED adsp snps TO BE COMPATIBLE WITH --export-allele ##########
  tmp_pos_file = data.frame(ID = ukb_snp$bim$ID, REF = ukb_snp$bim$REF)
  fwrite(tmp_pos_file, sep = ' ',
         paste0(adsp_dir,'tmp_rs_file.txt'),col.names = F)
  #############################################################################

  adsp_snp_path = paste0('/home/panwei/he000176/deepRIV/ImputedTraits/ADSP/adsp_plink_chr',chr)
  adsp_out = paste0(adsp_dir,gene_name)

  adsp_plink_command = paste0('module load plink/2.00-alpha-091019; plink2 --bfile ',
                              adsp_snp_path,
                             ' --chr ', chr,
                             ' --extract ',adsp_dir,'tmp_rs_file.txt',
                             ' --export A include-alt',
                             ' --export-allele ', adsp_dir, 'tmp_rs_file.txt',
                             ' --snps-only just-acgt',
                             ' --out ',adsp_out)
  adsp_plink_msg = system(adsp_plink_command,intern=TRUE)
  
  adsp_snp = list()
  adsp_snp$bed = tryCatch(fread(paste0(adsp_out,'.raw'), data.table=F),
                     error=function(e){cat("ERROR :",
                                           conditionMessage(e),
                                           "\n")})
  if(length(adsp_snp)==0)
  {
    FINAL_RESULT = list('NoValue')
    return(FINAL_RESULT)
  }

  adsp_snp$fam = adsp_snp$bed[,c(1:6)] 
  # create bim file without pos
  tmp_adsp_names = colnames(adsp_snp$bed)[-c(1:6)]
  adsp_snp$bim = data.frame(ID = sub("_.*", "", tmp_adsp_names), 
                            REF = sub(".*_(.*)\\(/.*", "\\1", tmp_adsp_names), 
                            ALT = sub(".*\\(/(.*)\\)", "\\1", tmp_adsp_names))
  # check for special characters in rs id, REF and ALT alleles
  unique(unlist(strsplit(paste(adsp_snp$bim$ID,collapse=''),split='')))
  unique(unlist(strsplit(paste(adsp_snp$bim$REF,collapse=''),split='')))
  unique(unlist(strsplit(paste(adsp_snp$bim$ALT,collapse=''),split='')))

  # turn adsp_snp$bed into an R matrix
  adsp_snp$bed = as.matrix(adsp_snp$bed[,-c(1:6)])
  colnames(adsp_snp$bed) = adsp_snp$bim$ID
  rownames(adsp_snp$bed) = adsp_snp$fam$IID 

  unlink(adsp_dir, recursive = TRUE)
  rm(tmp_adsp_names)
  
  ########################################################################
  gtex_bim_tmp = merge(snp$bim,gtex_dic,by.x='ID',by.y='V1',sort=F)
  all(snp$bim$ID==gtex_bim_tmp$ID)
  snp$bim$ID = gtex_bim_tmp$V7
  colnames(snp$bed) = snp$bim$ID

  # remove ambiguous (palindromic) SNPs and flip alleles (ref, effect) for UKB
  snp_bim_ukb = merge(snp$bim,ukb_snp$bim,by=c("ID"))
  remove_flip = allele.qc(snp_bim_ukb$REF.x,snp_bim_ukb$ALT.x,
                          snp_bim_ukb$REF.y,snp_bim_ukb$ALT.y)
  print(paste0("any SNP's orientation needs to be flipped? ", any(remove_flip$flip[remove_flip$keep]))) # should be FALSE
  flip_snp = snp_bim_ukb$ID[remove_flip$flip]
  snp_bim_ukb = snp_bim_ukb[remove_flip$keep,]
  
  ukb_snp_bed = ukb_snp$bed[,which(colnames(ukb_snp$bed) %in% snp_bim_ukb$ID),drop=FALSE] 
  ukb_snp_bed[,which(colnames(ukb_snp_bed) %in% flip_snp)] = 2 - ukb_snp_bed[,which(colnames(ukb_snp_bed) %in% flip_snp)]
  
  # remove ambiguous (palindromic) SNPs and flip alleles (ref, effect) for adsp
  snp_bim_adsp = merge(snp$bim,adsp_snp$bim,by=c("ID"))
  remove_flip = allele.qc(snp_bim_adsp$REF.x,snp_bim_adsp$ALT.x,
                          snp_bim_adsp$REF.y,snp_bim_adsp$ALT.y)
  print(paste0("any SNP's orientation needs to be flipped? ", any(remove_flip$flip[remove_flip$keep]))) # should be FALSE
  flip_snp = snp_bim_adsp$ID[remove_flip$flip]
  snp_bim_adsp = snp_bim_adsp[remove_flip$keep,]
  adsp_snp_bed = adsp_snp$bed[,which(colnames(adsp_snp$bed) %in% snp_bim_adsp$ID),drop=FALSE] 
  adsp_snp_bed[,which(colnames(adsp_snp_bed) %in% flip_snp)] = 2 - adsp_snp_bed[,which(colnames(adsp_snp_bed) %in% flip_snp)]
  
  # common snps
  common_snps = intersect(colnames(snp$bed),colnames(adsp_snp_bed))
  common_snps = intersect(common_snps, colnames(ukb_snp_bed))
  
  snp_bed_colname = colnames(snp$bed)
  snp_bed_ind = match(common_snps, snp_bed_colname)
  SNP_BED = snp$bed[, snp_bed_ind, drop = FALSE]
  
  ukb_bed_colname = colnames(ukb_snp_bed)
  ukb_bed_ind = match(common_snps, ukb_bed_colname)
  ukb_snp_bed = ukb_snp_bed[, ukb_bed_ind, drop = FALSE]
  
  adsp_bed_colname = colnames(adsp_snp_bed)
  adsp_bed_ind = match(common_snps, adsp_bed_colname)
  adsp_snp_bed = adsp_snp_bed[, adsp_bed_ind, drop = FALSE]
  
  # match adsp_pheno with adsp_snp_bed
  ### NEW replace : with _ and only remove the leaning 0_ ###
  rownames(adsp_snp_bed) = sub("^0_", "", rownames(adsp_snp_bed))
  ######

  keep_ukb_indiv = intersect(adsp_pheno$id,adsp_snp$fam$IID)
  adsp_pheno = adsp_pheno %>% filter(id %in% keep_ukb_indiv)
  adsp_snp_bed = adsp_snp_bed[which(rownames(adsp_snp_bed)%in%keep_ukb_indiv),,drop=FALSE]
  adsp_pheno = adsp_pheno[match(rownames(adsp_snp_bed),adsp_pheno$id),,drop=FALSE]
  
  cov5table = cov5table[match(rownames(SNP_BED),cov5table$ID),,drop=FALSE]
  
  keep_ukb_indiv = intersect(ukb_pheno$f.eid,ukb_snp$fam$IID)
  ukb_pheno = ukb_pheno %>% filter(f.eid %in% keep_ukb_indiv)
  ukb_snp_bed = ukb_snp_bed[which(rownames(ukb_snp_bed)%in%keep_ukb_indiv),,drop=FALSE]
  ukb_pheno = ukb_pheno[match(rownames(ukb_snp_bed),ukb_pheno$f.eid),,drop=FALSE]
  # ukb_snp_bed = mean_imputation(ukb_snp_bed)
  
  if(nrow(ukb_pheno)< 3){
    FINAL_RESULT = list('NoValue')
    real_data_result = c(real_data_result,list(FINAL_RESULT = FINAL_RESULT))
    next()
  }
  if(length(snp_bed_ind) < 2)
  {
    FINAL_RESULT = list('NoValue')
    return(FINAL_RESULT)
  }
  ### prune
  cor_cutoff = 0.8
  cor_bed = abs(cor(SNP_BED))
  cor_bed = (cor_bed < cor_cutoff)^2
  diag(cor_bed) = 1
  i = 1
  while(i < nrow(cor_bed) )
  {
    ind = which(cor_bed[i,] == 1)
    cor_bed = as.matrix(cor_bed[ind,ind])
    i = i + 1
  }
  if(nrow(cor_bed) == 1)
  {
    FINAL_RESULT = list('NoValue')
    return(FINAL_RESULT)
  }
  ind = which(is.element(colnames(SNP_BED),colnames(cor_bed)))
  SNP_BED = SNP_BED[,ind,drop=F]
  ukb_snp_bed = ukb_snp_bed[,ind,drop=F]
  adsp_snp_bed = adsp_snp_bed[,ind,drop=F]
  snp_bim_ukb = snp_bim_ukb[ind,,drop=F]
  
  ### regress gene exp on covariates
  X = as.matrix(gene_exp[1,-(1:4)])
  X = X[,match(rownames(SNP_BED),colnames(X)),drop=F]
  X1 = scale(as.numeric(X))
  
  lm_RegCov = lm(X1 ~.,data = data.frame(X1,cov5table[,-c(1)]))
  X1 = lm_RegCov$residuals
  X1 = scale(X1)
  
  X2 = X1^2
  lm_RegCov = lm(X2 ~.,data = data.frame(X2,cov5table[,-c(1,2)]))
  X2 = lm_RegCov$residuals
  X2 = scale(X2)
  
  if(ncol(SNP_BED) > 50)
  {
    ind1 = order(abs(cor(X1,SNP_BED)),decreasing = T)[1:50]
    ind2 = order(abs(cor(X2,SNP_BED)),decreasing = T)[1:50]
    ind = union(ind1,ind2)
    
    ind = sort(ind)
    SNP_BED = SNP_BED[,ind,drop=F]
    ukb_snp_bed = ukb_snp_bed[,ind,drop=F]
    adsp_snp_bed = adsp_snp_bed[,ind,drop=F]
    snp_bim_ukb = snp_bim_ukb[ind,]
  }
  SNP_BED = scale(SNP_BED)
  # ukb_snp_bed = scale(ukb_snp_bed)
  
  ###
  lm_stage1_X1 = lm(X1 ~ SNP_BED)
  hatbetaX1 = lm_stage1_X1$coefficients[-1]
  na_ind = which(!is.na(hatbetaX1))
  check_ukb = apply(ukb_snp_bed,2,sd)
  # na_ind_ukb = which(!is.na(check_ukb))
  # na_ind = intersect(na_ind,na_ind_ukb)
  SNP_BED = SNP_BED[,na_ind,drop=FALSE]
  ukb_snp_bed = ukb_snp_bed[,na_ind,drop=FALSE]
  adsp_snp_bed = adsp_snp_bed[,na_ind,drop=F]
  
  lm_stage1_X1 = 
    step(lm(X1~.,data = data.frame(X1,SNP_BED)),direction = "backward",trace=FALSE)
  AIC_stage1_X1 = AIC(lm_stage1_X1)
  BIC_stage1_X1 = BIC(lm_stage1_X1)
  lm_stage1_X1 = summary(lm_stage1_X1)
  stage1_sigma = lm_stage1_X1$sigma
  rsq_stage1_X1 = lm_stage1_X1$r.squared
  adjrsq_stage1_X1 = lm_stage1_X1$adj.r.squared
  se_betaX1 = hatbetaX1 = rep(0,ncol(SNP_BED))
  coef_X1 = lm_stage1_X1$coefficients
  name_coef_X1 = substr(rownames(coef_X1),1,30)
  name_SNP_BED = colnames(SNP_BED)
  for(beta_ind in 1:nrow(coef_X1))
  {
    ii = which(name_SNP_BED == name_coef_X1[beta_ind])
    hatbetaX1[ii] = coef_X1[beta_ind,1]
    se_betaX1[ii] = coef_X1[beta_ind,2]
  }
  
  num_snps = sum(hatbetaX1 != 0)
  
  ## get E(X2 | Z)
  lm_stage1_X2 = 
    step(lm(X2~.,data = data.frame(X2,SNP_BED)),direction = "backward",trace=FALSE)
  AIC_stage1_X2 = AIC(lm_stage1_X2)
  BIC_stage1_X2 = BIC(lm_stage1_X2)
  lm_stage1_X2 = summary(lm_stage1_X2)
  rsq_stage1_X2 = lm_stage1_X2$r.squared
  adjrsq_stage1_X2 = lm_stage1_X2$adj.r.squared
  se_betaX2 = hatbetaX2 = rep(0,ncol(SNP_BED))
  coef_X2 = lm_stage1_X2$coefficients
  stage1_X2_fstat = lm_stage1_X2$fstatistic[1]
  name_coef_X2 = substr(rownames(coef_X2),1,30)
  name_SNP_BED = colnames(SNP_BED)
  for(beta_ind in 1:nrow(coef_X2))
  {
    ii = which(name_SNP_BED == name_coef_X2[beta_ind])
    hatbetaX2[ii] = coef_X2[beta_ind,1]
    se_betaX2[ii] = coef_X2[beta_ind,2]
  }
  
  
  if(sum(abs(hatbetaX1)) == 0 || num_snps < 2)
  {
    FINAL_RESULT = list('NoValue')
    return(FINAL_RESULT)
  }
  
  if(nrow(ukb_snp_bed) < 500){
    FINAL_RESULT = list('NoValue')
    return(FINAL_RESULT)
  }
  
  if(is.null(lm_stage1_X1$fstatistic)){
    stage1_fstat = 'NoValue'
    stage1_pval = 'NoValue'
    FINAL_RESULT = list('NoValue')
    return(FINAL_RESULT)
  }else if(lm_stage1_X1$fstatistic[1] < 10){
    FINAL_RESULT = list('NoValue')
    return(FINAL_RESULT)
  }else{
    stage1_fstat = lm_stage1_X1$fstatistic[1]
    stage1_pval = pf(lm_stage1_X1$fstatistic[1],lm_stage1_X1$fstatistic[2],
                     lm_stage1_X1$fstatistic[3], lower.tail = F)
  }
  
  ### UKB
  # ukb_pheno[,2] = scale(ukb_pheno[,2])
  # tmp_ukb_snp_bed = ukb_snp_bed
  # for(i in 1:ncol(tmp_ukb_snp_bed)){
  #   idx = which(is.na(tmp_ukb_snp_bed[,i]))
  #   tmp_ukb_snp_bed[idx,i] = mean(tmp_ukb_snp_bed[,i], na.rm = T)
  # }
  # Xhat_ukb_total = scale(tmp_ukb_snp_bed) %*% hatbetaX1 + coef_X1[1,1]
  y_ukb_ID = as.numeric(ukb_pheno$f.eid)
  y_ukb_total = as.numeric(unlist(ukb_pheno[,2]))
  rs_id = colnames(ukb_snp_bed)
  y_adsp_ID = rownames(adsp_snp_bed)
  y_adsp_total = as.numeric(unlist(adsp_pheno[,2]))
  # stage2_test = lm(y_ukb_total~Xhat_ukb_total)
  # stage2_test = summary(stage2_test)

  if (save_models){
    fwrite(ukb_snp_bed, file = z_path)
    fwrite(as.data.frame(hatbetaX1), file = beta_path)
    fwrite(data.frame(theta = stage2_test$coefficients[2,1]), file = theta_path)
    if (!is.null(y_path)){
      fwrite(data.frame(y = y_ukb_total), file = y_path)
    }
  }
  
  FINAL_RESULT = list(gene=gene_name,
                      chr=chr,
                      # Xhat_ukb_total = Xhat_ukb_total,
                      ukb_snp_bed = ukb_snp_bed,
                      intercept = coef_X1[1,1],
                      hatbetaX1 = hatbetaX1,
                      intercept2 = coef_X2[1,1],
                      hatbetaX2 = hatbetaX2,
                      y_ukb_ID = y_ukb_ID,
                      stage1_fstat = stage1_fstat,
                      stage1_pval = stage1_pval,
                      stage1_X2_fstat = stage1_X2_fstat,
                      num_snps = num_snps,
                      stage1_sigma = stage1_sigma,
                      y_ukb_total = y_ukb_total,
                      rs_id = rs_id,
                      adsp_snp_bed = adsp_snp_bed,
                      y_adsp_ID = y_adsp_ID,
                      y_adsp_total = y_adsp_total,
                      tmp_dir_list = tmp_dirs_list
                      
  )
  return(FINAL_RESULT)
}

.Last <- function() {
  for (dir in tmp_dirs_list) {
    tryCatch({
      unlink(dir, recursive = TRUE, force = TRUE)
      system(paste0('rm -rf ',dir))
      if (!dir.exists(dir)) {
        message("Temporary directory successfully removed on exit: ", dir)
      } else {
        warning("Failed to remove temporary directory on exit: ", dir)
      }
    }, error = function(e) {
      warning("Error during .Last execution for directory ", dir, ": ", e$message)
    })
  }
}
print(.Last)




#   plink_command = paste0("module load plink; \n","plink --bfile /home/panwei/he000176/deepRIV/UKB/data/GTEx/GTEx_Analysis_2017-06-05_v8_WholeGenomeSeq_838Indiv_Analysis_Freeze.SHAPEIT2_phased_MAF01",
#                          " --chr ",chr," --from-kb ",
#                          start," --to-kb ",end,
#                          " --keep ",whole_blood_buddy,
#                          " --geno 0 --maf 0.05 --hwe 0.001  ",
#                          " --make-bed --out ",GTEx_dir,
#                          gene_name,sep="")
#   plink_msg=system(plink_command,intern=TRUE)



#   tmp_pos_file = data.frame(chr = chr, frombp = snp$bim$V4,
#                             tobp = snp$bim$V4, id = snp$bim$V2)
#   fwrite(tmp_pos_file, sep = ' ',
#          paste0(adsp_dir,'tmp_pos_file.txt'),col.names = F)
#   adsp_plink_command = paste0('module load plink/2.00-alpha-091019; plink2 --bfile ',
#                               adsp_snp_path,
#                              ' --chr ', chr,
#                              ' --extract range ',adsp_dir,'tmp_pos_file.txt',
#                              #' --maf 0.05 --hwe 0.001 ',
#                              ' --make-bed',
#                              ' --out ',adsp_out)
  
#   adsp_plink_msg = system(adsp_plink_command,intern=TRUE)