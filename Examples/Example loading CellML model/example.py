import sabs_pkpd
import myokit
import myokit.formats.cellml
import myokit.formats


importer = myokit.formats.importer('cellml')
model = importer.model('C:/Users/yanral/Documents/CellML models/11.2019/Conudctances comparison/grandi_pasqualini_bers_2010.cellml')

model, protocol = sabs_pkpd.load_model.convert_protocol(model)
myokit.save('./grandi_pasqualini_bers_2010.mmt', model=model, protocol=protocol)
