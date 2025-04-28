#ifndef _GMX_LIBRARY_CODE
#define _GMX_LIBRARY_CODE

#include <gromacs/trajectoryanalysis/topologyinformation.h>
#include <gromacs/fileio/tpxio.h>
#include <gromacs/math/vec.h>
#include <gromacs/pbcutil/pbc.h>
#include <gromacs/fileio/confio.h>
#include <gromacs/topology/index.h>

#include <vector>

namespace resdata::gmx
{
  static inline void mtopGetMolblockIndex(const gmx_mtop_t &mtop,
                                          int globalAtomIndex,
                                          int *moleculeBlock,
                                          int *moleculeIndex,
                                          int *atomIndexInMolecule)
  {
    // GMX_ASSERT(globalAtomIndex >= 0, "The atom index to look up should not be negative");
    // GMX_ASSERT(globalAtomIndex < mtop.natoms, "The atom index to look up should be within range");
    // GMX_ASSERT(moleculeBlock != nullptr, "molBlock can not be NULL");
    // GMX_ASSERT(!mtop.moleculeBlockIndices.empty(), "The moleculeBlockIndices should not be empty");
    // GMX_ASSERT(*moleculeBlock >= 0,
    //  "The starting molecule block index for the search should not be negative");
    // GMX_ASSERT(*moleculeBlock < gmx::ssize(mtop.moleculeBlockIndices),
    //  "The starting molecule block index for the search should be within range");

    /* Search the molecule block index using bisection */
    int molBlock0 = -1;
    int molBlock1 = mtop.molblock.size();

    int globalAtomStart = 0;
    while (TRUE)
    {
      globalAtomStart = mtop.moleculeBlockIndices[*moleculeBlock].globalAtomStart;
      if (globalAtomIndex < globalAtomStart)
      {
        molBlock1 = *moleculeBlock;
      }
      else if (globalAtomIndex >= mtop.moleculeBlockIndices[*moleculeBlock].globalAtomEnd)
      {
        molBlock0 = *moleculeBlock;
      }
      else
      {
        break;
      }
      *moleculeBlock = ((molBlock0 + molBlock1 + 1) >> 1);
    }

    int molIndex = (globalAtomIndex - globalAtomStart) / mtop.moleculeBlockIndices[*moleculeBlock].numAtomsPerMolecule;
    if (moleculeIndex != nullptr)
    {
      *moleculeIndex = molIndex;
    }
    if (atomIndexInMolecule != nullptr)
    {
      *atomIndexInMolecule = globalAtomIndex - globalAtomStart - molIndex * mtop.moleculeBlockIndices[*moleculeBlock].numAtomsPerMolecule;
    }
  }
  void mtopGetAtomAndResidueName(const gmx_mtop_t &mtop,
                                 int globalAtomIndex,
                                 int *moleculeBlock,
                                 const char **atomName,
                                 int *residueNumber,
                                 const char **residueName,
                                 int *globalResidueIndex)
  {
    int moleculeIndex = 0;
    int atomIndexInMolecule = 0;
    mtopGetMolblockIndex(mtop, globalAtomIndex, moleculeBlock, &moleculeIndex, &atomIndexInMolecule);

    const gmx_molblock_t &molb = mtop.molblock[*moleculeBlock];
    const t_atoms &atoms = mtop.moltype[molb.type].atoms;
    const MoleculeBlockIndices &indices = mtop.moleculeBlockIndices[*moleculeBlock];
    if (atomName != nullptr)
    {
      *atomName = *(atoms.atomname[atomIndexInMolecule]);
    }
    if (residueNumber != nullptr)
    {
      if (atoms.nres > mtop.maxResiduesPerMoleculeToTriggerRenumber())
      {
        *residueNumber = atoms.resinfo[atoms.atom[atomIndexInMolecule].resind].nr;
      }
      else
      {
        /* Single residue molecule, keep counting */
        *residueNumber = indices.residueNumberStart + moleculeIndex * atoms.nres + atoms.atom[atomIndexInMolecule].resind;
      }
    }
    if (residueName != nullptr)
    {
      *residueName = *(atoms.resinfo[atoms.atom[atomIndexInMolecule].resind].name);
    }
    if (globalResidueIndex != nullptr)
    {
      *globalResidueIndex = indices.globalResidueStart + moleculeIndex * atoms.nres + atoms.atom[atomIndexInMolecule].resind;
    }
  }
} // namespace resdata::gmx

#endif // _GMX_LIBRARY_CODE