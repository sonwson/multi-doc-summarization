import mongoose from 'mongoose';

const historySchema = new mongoose.Schema(
  {
    user: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'User',
      required: true
    },
    inputs: {
      type: [String],
      required: true,
      validate: [(value) => value.length > 0, 'At least one input is required']
    },
    summary: {
      type: String,
      required: true
    },
    sourceFiles: {
      type: [String],
      default: []
    }
  },
  { timestamps: true }
);

export const History = mongoose.model('History', historySchema);
