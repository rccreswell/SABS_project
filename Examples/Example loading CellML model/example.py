import sabs_pkpd
import myokit
import myokit.formats.cellml
import myokit.formats


cellml = 'C:/Users/yanral/Documents/PhD/CellML models/11.2019/'\
         'Conudctances comparison/grandi_pasqualini_bers_2010.cellml'
mmt = 'C:/Users/yanral/Documents/PhD/2D maps/.mmt models/'\
      'grandi_pasqualini_bers_2010.mmt'

model, protocol = sabs_pkpd.load_model.load_model_from_cellml(cellml, mmt)
