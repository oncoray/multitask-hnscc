library(survcomp)

compute_cindex = function(data, pred_col, time_col, event_col){
  x = data[[pred_col]]
  time = data[[time_col]]
  event = data[[event_col]]
  weights = rep(1, dim(data)[1])
  #strat = rep(1, dim(data)[1])

  cindex = concordance.index(x, surv.time=time, surv.event=event,
                             #cl=,
                             weights=weights,
                             #strat=strat,
                             alpha=0.05, method="noether")
  return(cindex)
}


workdir = "/path/to/the/results/of/your/experiments"

tasks = c()
backbones = c()
segs = c()
densenets = c()
cohorts = c()
heads = c()
cs = c()
lowers = c()
uppers = c()
ps = c()

for(task in c("multitask", "st-cox", "st-gh")){
  for(backbone in c("cnn", "vit")){
    for(seg in c("with", "without")){
      for(dn in c("with", "without")){
        dirname = paste(task, "+", backbone, "_", seg, "-seg_", dn, "-densenet", sep="")
        file_path = file.path(workdir, dirname)
        ensemble_dir = list.files(file_path, pattern="ensemble_mean", recursive=TRUE, include.dirs=TRUE, full.names=TRUE)[[1]]
        print(paste(task, " ", backbone, " ", seg, "-seg ", dn, "-densenet", sep=""))
        for(cohort in list.files(ensemble_dir, full.names=TRUE, include.dirs=TRUE)){
          if(!dir.exists(cohort)){next}
          cat(paste("\t", basename(cohort), "\n"))
          for(survhead in list.files(cohort, "*_predictions.csv", full.names=TRUE)){
            headname = strsplit(basename(survhead), "_")[[1]][1]
            # read the file
            preds = read.csv(survhead)
            if(headname == "gensheimer"){
              pred_col =  "predicted_survival_for_cindex_time_24.0"
              headname = paste(headname, "24-months", sep="_")
            }
            else if(headname == "cox"){
              pred_col = "ensemble_prediction"

            }
            cindex = compute_cindex(preds, pred_col=pred_col, time_col="event_time", event_col="event")
            #cindices[[f]] = cindex

            c = 1. - cindex$c.index
            l = 1. - cindex$lower
            u = 1. - cindex$upper
            if(l >= u){
              tmp = l
              l = u
              u = tmp
            }
            pval = cindex$p.value

            cat(paste("\t\t", headname, "\n"))
            cat(paste("\t\t\tc=", round(c, 3), "(", round(l, 3), "-", round(u, 3), ")", "| p(c==0.5)=", round(pval, 3), "\n"))

            tasks = append(tasks, task)
            backbones = append(backbones, backbone)
            segs = append(segs, seg)
            densenets = append(densenets, dn)
            cohorts = append(cohorts, basename(cohort))
            heads = append(heads, headname)
            cs = append(cs, c)
            lowers = append(lowers, l)
            uppers = append(uppers, u)
            ps = append(ps, pval)

          }
        }
        cat("\n")
      }
    }
  }
}

data = data.frame(task=tasks, backbone=backbones, seg=segs, densenet=densenets, cohort=cohorts, survival_head=heads, c=cs, lower=lowers, upper=uppers, p=ps)
write.table(data, file="ensemble_results_with_confidence_intervals.csv", sep=",", row.names=FALSE)